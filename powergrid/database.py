# Attention: This database is only for optimal power flow
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
        self.base_voltage = 22        # base voltage (kV)
        self.base_power = 10          # base power (MVA)
        self.base_impedance = self.base_voltage ** 2 / self.base_power
        self.slack = 1
        self.power_factor = 0.95      # load power factor (lagging)
        # construct power grid
        self.net = pp.create_empty_network(name="Norwegian22kV", add_stdtypes=False)
        self.__set_std_type()           # add standard line type
        self.__set_power_grid()         # construct power grid by pandapower
        # get load data
        self.load_peak_data = pd.read_csv(os.path.join(current_dir, "data/Norwegian/peak_loads.csv")).set_index('index')
        self.load_data = pd.read_csv(os.path.join(current_dir, "data/Norwegian/loads.csv")).set_index('index')
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
                vn_kv=self.base_voltage, 
                index=int(row['index']), 
                geodata=(float(row['x']), float(row['y'])),
                max_vm_pu=float(row['Vmax']), 
                min_vm_pu=float(row['Vmin']),
                in_service=True
            )
            # Main Feeder
            if int(row["type"]) == 3:
                s = pp.create_ext_grid(self.net, bus=int(row["index"]), vm_pu=1.0, va_degree=0.0)
                pp.create_poly_cost(self.net, element=s, et="ext_grid", cp1_eur_per_mw=5.0)
        for _, row in pd_branch.iterrows():
            pp.create_line(
                net=self.net, 
                from_bus=int(row['from']), 
                to_bus=int(row['to']), 
                length_km=float(row['length_km']),
                std_type=row['type'], 
                index=int(row['index']),
                in_service=True
            )
    
    # networkx format
    def __get_topology(self):
        pd_bus = pd.read_csv(os.path.join(current_dir, "data/Norwegian/bus.csv"))
        pd_branch = pd.read_csv(os.path.join(current_dir, "data/Norwegian/branch.csv"))
        G = nx.DiGraph()
        for _, row in pd_bus.iterrows():
            G.add_node(
                int(row['index']),
                pos=(float(row['x']), float(row['y'])),
                Vmax=float(row["Vmax"]),
                Vmin=float(row["Vmin"]),
                color='lightblue'
            )
        for _, row in pd_branch.iterrows():
            G.add_edge(
                int(row['from']),
                int(row['to']),
                R=float(row["R"]),
                X=float(row["X"]),
                B=float(row["B"]),
                rate_A=float(row["rate_A"]),
                length=float(row["length_km"])
            )
        # special node
        G.nodes[1]['color'] = 'tab:red'     # Mian feeder
        G.nodes[36]['color'] = 'tab:blue'   # Backup feeder1
        G.nodes[62]['color'] = 'tab:blue'   # Backup feeder2
        G.nodes[88]['color'] = 'tab:blue'   # Backup feeder3
        for n in G.nodes():                 # hilight load node
            if n in self.load_nodes["exist"]:
                G.nodes[n]["color"] = 'tab:green'
        for n in G.nodes():
            b1 = 0
            for i in G.predecessors(n):
                b1 += G.edges[(i, n)]['B']
            b2 = 0
            for i in G.successors(n):
                b2 += G.edges[(n, i)]['B']
            G.nodes[n]['b'] = (b1 + b2) / 2.0
        return G
    
    # get load node list
    def __get_load_nodes(self):
        nodes = {}
        nodes["exist"] = self.load_data.columns.tolist()
        nodes["plan"] = self.load_plan_data.columns.tolist()
        return nodes

    # export power grid by pandapower format(json)
    def export_network(self):
        pp.to_json(self.net, "Norwegian.json")
        return
    
    # export power grid by networkx format(gml)
    def export_topology(self):
        nx.write_gml(self.topology, "Norwegian.gml")
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
        plt.close()
    
    # example of power flow
    def example_power_flow(self):
        net = copy.deepcopy(self.net)
        # get load data (example: peak load)
        pd_load = self.load_peak_data
        for index, row in pd_load.iterrows():
            pp.create_load(net, bus=int(index), p_mw=float(row["p_mw"]), q_mvar=float(row["q_mvar"]))
        # calculate
        pp.runpp(net)
        # print
        net.res_bus.to_csv("example_pf_bus_sol.csv")
        net.res_line.to_csv("example_pf_branch_sol.csv")
        return

    def __get_reactive_power(self, p: float):
        return math.sqrt((p / self.power_factor) ** 2 - p ** 2)

    def get_peak_load_data(self):
        pd_load = self.load_peak_data
        load = {}
        for index, row in pd_load.iterrows():
            load[int(index)] = (
                float(row["p_mw"]) / self.base_power,
                float(row['q_mvar']) / self.base_power
            )
        return load

    def get_load_data(self, load_time=0):
        if load_time >= len(self.load_data.index) or load_time < 0:
            raise ValueError('illegal load_time!')
        load = {}
        for col in self.load_data.columns:
            load[int(col)] = (
                self.load_data[col][load_time] / self.base_power,
                self.__get_reactive_power(self.load_data[col][load_time]) / self.base_power
            )
        return load

    def get_all_load_data(self):
        load = {}
        for col in self.load_data.columns:
            load[int(col)] = []
        for col in self.load_data.columns:
            for i in self.load_data.index:
                load[int(col)].append(
                    self.load_data[col][i] / self.base_power,
                    self.__get_reactive_power(self.load_data[col][i]) / self.base_power
                )
        return load

    def run_powerflow(self, load_time=0):
        if load_time >= len(self.load_data.index) or load_time < 0:
            raise ValueError('illegal load_time!')
        net = copy.deepcopy(self.net)
        for index, row in self.load_peak_data.iterrows():
            pp.create_load(net, bus=int(index), index=int(index), p_mw=float(row["p_mw"]), q_mvar=float(row["q_mvar"]))
        for col in self.load_data.columns:
            net.load.at[int(col), "p_mw"] = float(self.load_data[col][load_time])
            net.load.at[int(col), "q_mvar"] = float(self.__get_reactive_power(self.load_data[col][load_time]))
        pp.runopp(net)
        net.res_bus.to_csv(f'res_bus_{load_time}.csv')
        net.res_line.to_csv(f'res_branch_{load_time}.csv')



""" IEEE 33 bus power grid
The data origin is the paper M. Baran, F. Wu, 
“Network reconfiguration in distribution systems for loss reduction and load balancing”
IEEE Transactions on Power Delivery, 1989.
"""
class IEEE33:
    def __init__(self):
        self.base_voltage = 12.66     # kV
        self.base_power = 10          # MVA
        self.base_impedance = self.base_voltage ** 2 / self.base_power
        self.slack = 0
        self.grid = self.__get_topology()
        self.load = self.__get_load_data()
        self.cost = self.__get_cost_data()

    def __get_topology(self):
        pd_bus = pd.read_csv(os.path.join(current_dir, './data/IEEE33/bus.csv'))
        pd_branch = pd.read_csv(os.path.join(current_dir, './data/IEEE33/branch.csv'))
        G = nx.DiGraph()
        for _, row in pd_bus.iterrows():
            G.add_node(
                int(row['name']),
                Vmax=float(row['max_vm_pu']),
                Vmin=float(row['min_vm_pu']),
                g=0,
                b=0,
                pos=(float(row['geo_x']), float(row['geo_y']))
            )
        for _, row in pd_branch.iterrows():
            if row['in_service'] is True:
                G.add_edge(
                    int(row['from_bus']),
                    int(row['to_bus']),
                    R=float(row['r_ohm_per_km']) / self.base_impedance,
                    X=float(row['x_ohm_per_km']) / self.base_impedance
                )
        return G

    def __get_load_data(self):
        load = {}
        pd_load = pd.read_csv(os.path.join(current_dir, './data/IEEE33/load.csv'))
        for _, row in pd_load.iterrows():
            load[int(row['bus'])] = (
                float(row['p_mw']) / self.base_power,
                float(row['q_mvar']) / self.base_power
            )
        return load

    def __get_cost_data(self):
        cost = {}
        cost[0] = 20.0
        return cost
