import copy
import os
import math
import json

import pandas as pd
import pandapower as pp
import networkx as nx
import matplotlib.pyplot as plt


# global parameters
current_dir = os.path.dirname(__file__)


""" Norwegian medium voltage power distribution system
[1] I. B. Sperstad, O. B. Fosso, S. H. Jakobsen, A. O. Eggen, J. H. Evenstuen, and G. Kjølle, 
“Reference data set for a Norwegian medium voltage power distribution system,” 
Data in Brief, vol. 47, p. 109025, Apr. 2023, doi: 10.1016/j.dib.2023.109025.
Note: This database only support basic infomation of power grid, topology and load data.
"""
class Norwegian:
    def __init__(self):
        # the basic infomation of power grid
        self.__base_voltage = 22        # base voltage (kV)
        self.__base_power = 10          # base power (MVA)
        self.__power_factor = 0.95      # load power factor (lagging)
        # construct power grid
        self.net = pp.create_empty_network(name="Norwegian22kV", add_stdtypes=False)
        self.__set_std_type()           # add standard line type
        self.__set_power_grid()         # construct power grid by pandapower
        # get load data
        self.load_peak_data = pd.read_csv(os.path.join(current_dir, "data/Norwegian/peak_loads.csv")).set_index('index')
        self.load_data = self.__get_load_data()
        self.load_plan_data = pd.read_csv(os.path.join(current_dir, "data/Norwegian/loads_plan.csv")).set_index('index')
        self.load_nodes = self.__get_load_nodes()
        # set networkx data
        self.topology = self.__get_topology()

    def descriptions(self):
        with open(os.path.join(current_dir, 'data/Norwegian/descriptions.txt'), 'r', encoding='utf-8') as f:
            content = f.read()
            print(content)
        return
    
    # set standard line type by data
    def __set_std_type(self):
        pd_line = pd.read_csv(os.path.join(current_dir, "data/Norwegian/line_type.csv"))
        for _, data in pd_line.iterrows():
            line_data = {
                "c_nf_per_km": float(data["Cd_nF_per_km"]),
                "r_ohm_per_km": float(data["R_ohm_per_km"]),
                "x_ohm_per_km": float(data["X_ohm_per_km"]),
                "max_i_ka": float(data["Imax_A"]) / 1000
            }
            pp.create_std_type(self.net, line_data, data["type"], "line")
        return
    
    # construct power grid by pandapower
    def __set_power_grid(self):
        pd_bus = pd.read_csv(os.path.join(current_dir, "data/Norwegian/bus.csv"))
        pd_branch = pd.read_csv(os.path.join(current_dir, "data/Norwegian/branch.csv"))
        for _, row in pd_bus.iterrows():
            pp.create_bus(
                net=self.net, 
                vn_kv=self.__base_voltage, 
                index=int(row['index']), 
                geodata=(float(row['x']), float(row['y'])),
                max_vm_pu=float(row['Vmax']), 
                min_vm_pu=float(row['Vmin'])
            )
            # Main Feeder
            if int(row["type"]) == 3:
                pp.create_ext_grid(self.net, bus=int(row["index"]), vm_pu=1.0, va_degree=0.0)
        for _, row in pd_branch.iterrows():
            pp.create_line(
                net=self.net, 
                from_bus=int(row['from']), 
                to_bus=int(row['to']), 
                length_km=float(row['length_km']),
                std_type=row['type'], 
                index=int(row['index'])
            )
    
    # networkx format
    def __get_topology(self):
        pd_bus = pd.read_csv(os.path.join(current_dir, "data/Norwegian/bus.csv"))
        pd_branch = pd.read_csv(os.path.join(current_dir, "data/Norwegian/branch.csv"))
        G = nx.Graph()
        for _, row in pd_bus.iterrows():
            G.add_node(int(row['index']), pos=(float(row['x']), float(row['y'])), Vmax=float(row["Vmax"]), Vmin=float(row["Vmin"]), color='lightblue')
        for _, row in pd_branch.iterrows():
            G.add_edge(int(row['from']), int(row['to']), R=float(row["R"]), X=float(row["X"]), B=float(row["B"]),
                       rate_A=float(row["rate_A"]), length=float(row["length_km"]))
        # special node
        G.nodes[1]['color'] = 'tab:red'     # Mian feeder
        G.nodes[36]['color'] = 'tab:blue'   # Backup feeder1
        G.nodes[62]['color'] = 'tab:blue'   # Backup feeder2
        G.nodes[88]['color'] = 'tab:blue'   # Backup feeder3
        for n in G.nodes():                 # hilight load node
            if n in self.load_nodes["exist"]:
                G.nodes[n]["color"] = 'tab:green'
        return G
    
    # get load node list
    def __get_load_nodes(self):
        nodes = {}
        nodes["exist"] = self.load_data.columns.tolist()
        nodes["plan"] = self.load_plan_data.columns.tolist()
        return nodes
    
    # get load data
    def __get_load_data(self):
        origin_data = pd.read_csv(os.path.join(current_dir, "data/Norwegian/loads.csv")).set_index('index')
        peak_data = self.load_peak_data["p_mw"]
        for col in origin_data.columns:
            origin_data[col] = origin_data[col] * peak_data.at[int(col)]
        return origin_data

    # export power grid by pandapower format(json)
    def export_network(self):
        pp.to_json(self.net, "Norwegian.json")
        return
    
    # export power grid by networkx format(gml)
    def export_topology(self):
        nx.write_gml(self.topology, "Norwegian.gml")
        return
    
    # example of power flow
    def example_power_flow(self):
        net = copy.deepcopy(self.net)
        # get load data (example: peak load)
        pd_load = self.load_peak_data
        for _, row in pd_load.iterrows():
            pp.create_load(net, bus=int(row["index"]), p_mw=float(row["p_mw"]), q_mvar=float(row["q_mvar"]))
        # calculate
        pp.runpp(net)
        # print
        net.res_bus.to_csv("example_pf_bus_sol.csv")
        net.res_line.to_csv("example_pf_branch_sol.csv")
        return
    
    # show topology
    def show_topology(self, file_dir=None):
        plt.figure(figsize=(15, 6))
        pos = nx.get_node_attributes(self.topology, 'pos')
        colors = nx.get_node_attributes(self.topology, 'color').values()
        nx.draw(self.topology, pos=pos, with_labels=True, node_size=500, font_size=10, font_weight='bold', font_color='whitesmoke', node_color=colors)
        if file_dir == None:
            plt.show()
        else:
            plt.savefig(file_dir + "/Norwegian.png")
    
    # show load curve in the range of [l, r]
    def show_load_curve(self, n, l=0, r=8760):
        # wrong input parameter
        n = str(n)
        l = 0 if l > 8760 or l < 0 else l
        r = 8760 if r > 8760 or r < 0 else r
        # load data
        data_df = self.load_data
        plt.title(f"Exist Load Node {n}")
        if n in self.load_nodes["plan"]:
            data_df = self.load_plan_data
            plt.title(f"Plan Load Node {n}")
        elif n not in self.load_nodes["exist"]:
            print("There is no load at this node!")
            return
        series = data_df[n].tolist()[l: r]
        # paint
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.plot(list(range(l, r, 1)), series)
        plt.xlabel("hour")
        plt.ylabel("power (kW)")
        plt.show()
        return
    
    # initial the bus(node) result container of power flow
    def __init_pf_res_bus(self):
        res_bus = {
            "vm_pu": {},
            "va_degree": {},
            "p_mw": {},
            "q_mvar": {}
        }
        for i in self.net.bus.index:
            for k in res_bus.keys():
                res_bus[k][i] = []
        return res_bus

    # initial the branch(link) result container of power flow
    def __init_pf_res_branch(self):
        res_branch = {
            "p_from_mw": {},
            "q_from_mvar": {},
            "p_to_mw": {},
            "q_to_mvar": {},
            "pl_mw": {},
            "ql_mvar": {},
            "i_from_ka": {},
            "i_to_ka": {},
            "i_ka": {},
            "vm_from_pu": {},
            "va_from_degree": {},
            "vm_to_pu": {},
            "va_to_degree": {},
            "loading_percent": {}
        }
        for i in self.net.line.index:
            for k in res_branch.keys():
                res_branch[k][i] = []
        return res_branch

    # get power flow result in the range of [l, r]
    def pf_exist_load(self, l=0, r=8760, regen=False):
        l = 0 if l > 8760 or l < 0 else l
        r = 8760 if r > 8760 or r < 0 else r
        net = copy.deepcopy(self.net)
        res_bus = self.__init_pf_res_bus()
        res_branch = self.__init_pf_res_branch()
        res = {}
        file_path = os.path.join(current_dir, "./data/Norwegian/.storage/pf_result_all.json")
        if regen is False and os.path.exists(file_path):
            with open(file_path, 'r') as f:
                res = json.load(f)
            for k in res_bus.keys():
                for i in res_bus[k].keys():
                    res_bus[k][i] = res['bus'][k][str(i)][l: r]
            for k in res_branch.keys():
                for i in res_branch[k].keys():
                    res_branch[k][i] = res['branch'][k][str(i)][l: r]
        else:
            for index, row in self.load_peak_data.iterrows():
                pp.create_load(
                    net=net, 
                    index=int(index), 
                    bus=int(index), 
                    p_mw=float(row["p_mw"]), 
                    q_mvar=float(row["q_mvar"])
                )
            for i in range(l, r, 1):
                row = self.load_data.iloc[i]
                for col in self.load_data.columns:
                    p = row[col]
                    q = math.sqrt((p / self.__power_factor) ** 2 - p ** 2)
                    net.load.at[int(col), "p_mw"] = p
                    net.load.at[int(col), "q_mvar"] = q
                pp.runpp(net)
                # record result
                for index, row in net.res_bus.iterrows():
                    for col in res_bus.keys():
                        res_bus[col][index].append(row[col])
                for index, row in net.res_line.iterrows():
                    for col in res_branch.keys():
                        res_branch[col][index].append(row[col])
        res = {"bus": res_bus, "branch": res_branch}
        if l == 0 and r == 8760 and regen is True:
            with open(file_path, "w") as f:
                json.dump(res, f)
        with open("pf_result.json", 'w', encoding='utf-8') as f:
            json.dump(res, f)
        return
