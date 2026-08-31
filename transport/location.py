import copy
import networkx as nx
from gurobipy import Model, GRB, LinExpr, quicksum
from scipy.optimize import line_search
import numpy as np
import math
import json


""" add virtual nodes according to one OD pair.
@parameter: G, type: networkx(DiGraph), mean: extend traffic network
@parameter: R, type: 2-list[(o, d)],    mean: OD pairs
@return: None
"""
def __add_virtual_nodes__(G, R: list) -> None:
    for r in R:
        O = "O_" + str((r[0], r[1]))
        D = "D_" + str((r[0], r[1]))
        G.add_node(O)
        G.add_node(D)
        G.add_edge(O, r[0], d=0)
        G.add_edge(r[1], D, d=0)

# tool: set mapping between edge and its distance, its path of original network
def __path_to_edge__(edges, mapping, e, d, path):
    edges[e] = d
    mapping[e] = path

""" get all path which length is less than vehicle range M
@parameter: G, type: networkx(DiGraph), mean: original traffic network
@parameter: R, type: 2-list[(o, d)],    mean: OD pairs
@parameter: M, type: float,             mean: vehicle range
@return: edges, mapping
"""
def __get_edges_public__(G, M: float) -> tuple:
    public_edges = {}
    mapping = {}
    for n, (d, path) in nx.all_pairs_dijkstra(G, cutoff=M, weight='d'):
        for v in path.keys():
            if n == v:
                continue
            __path_to_edge__(public_edges, mapping, (n, v), d[v], path[v])
    return public_edges, mapping

""" get all path of virtual node which length is less than vehicle range M
@parameter: G, type: networkx(DiGraph), mean: original traffic network
@parameter: R, type: 2-list[(o, d)],    mean: OD pairs
@parameter: M, type: float,             mean: vehicle range
@return: edges, mapping
"""
def __get_edges_demand__(G, R: list, M: float) -> tuple:
    demand_edges = {}
    mapping = {}
    paths = dict(nx.all_pairs_dijkstra(G, cutoff=M, weight='d'))
    for r in R:
        O = "O_" + str((r[0], r[1]))
        D = "D_" + str((r[0], r[1]))
        for v in paths[r[0]][0].keys():
            if v == r[1]:
                __path_to_edge__(demand_edges, mapping, (O, D), paths[r[0]][0][r[1]], paths[r[0]][1][r[1]])
            __path_to_edge__(demand_edges, mapping, (O, v), paths[r[0]][0][v], paths[r[0]][1][v])
        for n in paths.keys():
            if r[1] in paths[n][0].keys():
                __path_to_edge__(demand_edges, mapping, (n, D), paths[n][0][r[1]], paths[n][1][r[1]])
    return demand_edges, mapping


# ====================================== fixed path model ======================================
class FixPath(object):
    """
    @parameter: ods,   type: dict-{index: demands},   mean: OD pairs and its demands
    @parameter: G,     type: networkx-DiGraph,        mean: traffic network
    @parameter: M,     type: float,                   mean: the range of electric vehicles
    @parameter: cost,  type: dict-{n: cost},          mean: the cost of charging station n
    @parameter: Paths, type: dict-{index: [path]},    mean: the paths, default: shortest path
    @warning: the index of ods must same to the index of Paths
    """
    def __init__(self, G, ods: dict, cost: dict, M: float, Paths=None):
        self.network = copy.deepcopy(G)     # original network
        self.M = M                          # vehicle range
        self.ods = ods                      # OD pairs and its demand size
        self.R = list(ods.keys())           # OD pairs
        self.cost = copy.deepcopy(cost)     # the cost of each node
        self.Paths = Paths
        if Paths is None:                   # the paths between OD pairs
            self.Paths = self.__get_all_OD_shortest_paths__()
        self.__A, self.__K = self.__preprocess__()
        # store results
        self.__stations = []
        self.__routes = {}
    
    def __init_results__(self):
        self.__stations = []
        self.__routes = {}

    # tool: get shortest path between O and D
    def __get_all_OD_shortest_paths__(self):
        paths = dict(nx.all_pairs_dijkstra_path(self.network, weight='d'))
        path_dict = {}
        for r in self.R:
            path_dict[r] = list(paths[r[0]][r[1]])
        return path_dict

    # tool: get the index of the first node covered subpath's ended point
    def __get_first_index__(self, path):
        d = 0
        index = 0
        for i in range(len(path) - 1):
            d += self.network.edges[(path[i], path[i + 1])]['d']
            if d > self.M:
                index = i
                break
        return index
    
    # tool: node n in path p covered edges
    def __covered_edges__(self, path, n):
        edges = []
        d = 0
        for i in range(n, len(path) - 1):
            d += self.network.edges[(path[i], path[i+1])]['d']
            if d > self.M:
                break
            edges.append((path[i], path[i+1]))
        return edges

    def __preprocess__(self):
        A = {}  # need covered subpath
        K = {}  # some arcs from the path can be covered by some nodes
        for p in self.Paths.keys():
            path = self.Paths[p]
            index = self.__get_first_index__(path)
            if index == 0:  # from O to D directly
                A[p] = []
                continue
            A[p] = path[index: ]
            # initial K_p: edge p in path can be covered some nodes
            K[p] = {}
            for i in range(len(A[p]) - 1):
                K[p][(A[p][i], A[p][i + 1])] = []
            for i in range(len(path) - 1):
                edges = self.__covered_edges__(path, i)
                for e in edges:
                    if e in K[p].keys():
                        K[p][e].append(path[i])
        return A, K

    # cover all path(/flow) minimize cost
    def opt_all_cover(self):
        self.__init_results__()
        m = Model()
        m.setParam('OutputFlag', 0)
        x = m.addVars(self.network.nodes(), vtype=GRB.BINARY, name="station")
        m.setObjective(quicksum(x[i] * self.cost[i] for i in self.network.nodes()), GRB.MINIMIZE)
        for r in self.R:
            path = self.__A[r]
            for p in range(len(path) - 1):
                expr = LinExpr()
                for i in self.__K[r][(path[p], path[p + 1])]:
                    expr += x[i]
                m.addConstr(expr >= 1, name="constr" + str(r))
        m.update()
        m.optimize()
        if m.status == GRB.Status.OPTIMAL:
            for i in self.network.nodes():
                if x[i].X == 1:
                    self.__stations.append(i)
            self.__routes = self.Paths
            return m.ObjVal
        else:
            return None
    
    # cover max flow limited by the cost of stations
    def opt_max_cover(self, total_cost):
        self.__init_results__()
        m = Model()
        m.setParam('OutputFlag', 0)
        x = m.addVars(self.network.nodes(), vtype=GRB.BINARY, name="station")
        y = m.addVars(self.R, vtype=GRB.BINARY, name='capture')
        m.setObjective(quicksum(self.ods[r] * y[r] for r in self.R), GRB.MAXIMIZE)
        for r in self.R:
            path = self.__A[r]
            for p in range(len(path) - 1):
                expr = LinExpr()
                for i in self.__K[r][(path[p], path[p + 1])]:
                    expr += x[i]
                m.addConstr(expr >= y[r], name="constr" + str(r))
        m.addConstr(x.sum() == total_cost)
        m.update()
        m.optimize()
        if m.status == GRB.Status.OPTIMAL:
            for i in self.network.nodes():
                if x[i].X == 1:
                    self.__stations.append(i)
            for r in self.R:
                if y[r].X == 1:
                    self.__routes[r] = self.Paths[r]
            return m.ObjVal
        return None
    
    # get results: stations
    def get_stations(self):
        return self.__stations
    
    # get results: routes
    def get_routes(self):
        return self.__routes
    
    # get results: edge flows
    def get_flows(self):
        flows = {}
        for e in self.network.edges():
            flows[e] = 0
        for k in self.ods.keys():
            path = self.__routes[k]
            for i in range(len(path) - 1):
                flows[(path[i], path[i + 1])] += self.ods[k]
        return flows


# ===================================== single path model =====================================
class SinglePath(object):
    """
    @parameter: R,    type: list-[(o, d)],    mean: OD demands
    @parameter: G,    type: networkx-DiGraph, mean: traffic network
    @parameter: M,    type: float,            mean: the range of electric vehicles
    @parameter: cost, type: dict-{n: cost},   mean: the cost of charging station n
    """
    def __init__(self, G, R: list[tuple], cost: dict, M: float):
        self.network = copy.deepcopy(G)     # original network
        self.M = M                          # vehicle range
        self.R = R                          # OD pairs
        self.cost = copy.deepcopy(cost)     # the cost of each node
        # preprocess
        self.mapping = {}                   # mapping: edge of extended network to path of original network
        self.inverse = {}                   # mapping: edge of original network to edge list of extended network
        self.__var_x = []                   # variables of extended network's edges
        self.G = nx.DiGraph()               # extended network
        G.add_nodes_from(self.network.nodes())
        self.__preprocess__()
        # Non-standard results
        self.__virtual_flow = {}
        # Standard results
        self.__stations = []
    
    def __init_results__(self):
        for r in self.R:
            self.__virtual_flow[r] = []
        self.__stations = []
    
    """ get variables from virtual edges
    @parameter: public_edges, type: dict,              mean: these edges belong to every demand r.
    @parameter: demnad_edges, type: dict,              mean: these edges belong to one demand.
    @return: variable list
    """
    def __get_variables__(self, public_edges: dict, demand_edges: dict) -> list:
        var_list = []
        for n, v in public_edges.keys():
            for r1, r2 in self.R:
                var_list.append((n, v, r1, r2))
        for r in self.R:
            O = "O_" + str((r[0], r[1]))
            D = "D_" + str((r[0], r[1]))
            for n, v in demand_edges.keys():
                if n == O or v == D:
                    var_list.append((n, v, r[0], r[1]))
        return var_list
    
    # tool: add virtual edges to G
    def __add_virtual_edges__(self, edges):
        for n, v in edges.keys():
            self.G.add_edge(n, v, d=edges[(n, v)])
    
    # get mapping: edge of original network to edge list of extended network
    def __inverse_mapping__(self) -> dict:
        inverse = {}
        # inital
        for e in self.network.edges():
            inverse[e] = []
        # interation
        for p in self.mapping.keys():
            for i in range(len(self.mapping[p]) - 1):
                inverse[(self.mapping[p][i], self.mapping[p][i+1])].append(p)
        return inverse

    def __preprocess__(self):
        __add_virtual_nodes__(self.G, self.R)
        public_edges, public_mapping = __get_edges_public__(self.network, self.M)
        demand_edges, demand_mapping = __get_edges_demand__(self.network, self.R, self.M)
        self.__var_x = self.__get_variables__(public_edges, demand_edges)
        self.__add_virtual_edges__(demand_edges)
        self.__add_virtual_edges__(public_edges)
        self.mapping.update(demand_mapping)
        self.mapping.update(public_mapping)
        self.inverse.update(self.__inverse_mapping__())

    """ optimal the location of cahrging station
    @parameter: num, type: int, mean: the number of station, default: None.
    @return: int, mean: the number of stations.
    """
    def opt(self, limits=None):
        self.__init_results__()
        m = Model()
        m.setParam('OutputFlag', 0)
        x = m.addVars(self.__var_x, vtype=GRB.BINARY)
        y = m.addVars(self.network.nodes(), vtype=GRB.BINARY)
        m.setObjective(quicksum(y[i] * self.cost[i] for i in self.network.nodes()), GRB.MINIMIZE)
        m.addConstrs((x.sum('O_' + str((r1, r2)), '*', r1, r2) - x.sum('*', 'O_' + str((r1, r2)), r1, r2) == 1 for r1, r2 in self.R),'origin')
        m.addConstrs((x.sum('D_' + str((r1, r2)), '*', r1, r2) - x.sum('*', 'D_' + str((r1, r2)), r1, r2) == -1 for r1, r2 in self.R),'destination')
        for i in self.network.nodes():
            m.addConstrs((x.sum(i, '*', r1, r2) - x.sum('*', i, r1, r2) == 0 for r1, r2 in self.R), 'edges')
            m.addConstrs((x.sum('*', i, r1, r2) <= y[i] for r1, r2 in self.R), 'station')
        if limits is not None:
            m.addConstr(y.sum() == limits, 'limits')
        m.update()
        m.optimize()
        if m.status == GRB.Status.OPTIMAL:
            for i in self.network.nodes():
                if y[i].X == 1:
                    self.__stations.append(i)
            for e1, e2, r1, r2 in self.__var_x:
                if x[e1, e2, r1, r2].X == 1:
                    self.__virtual_flow[(r1, r2)].append((e1, e2))
            return m.ObjVal
        return None
    
    # get results: stations
    def get_stations(self):
        return self.__stations
    
    # get results: routes
    def get_routes(self):
        routes = {}
        for r in self.__virtual_flow.keys():
            routes[r] = []
            # keep path connecting
            start_n = 'O_' + str((r[0], r[1]))
            ended_n = 'D_' + str((r[0], r[1]))
            while True:
                for e in self.__virtual_flow[r]:
                    if e[0] == start_n:
                        routes[r] += self.mapping[e] if e[1] == ended_n else self.mapping[e][:-1]
                        start_n = e[1]
                if start_n == ended_n:
                    break
        return routes
    
    def get_flows(self, ods):
        flows = {}
        for e in self.network.edges():
            flows[e] = 0
        routes = self.get_routes()
        for r in routes.keys():
            for i in range(len(routes[r]) - 1):
                flows[(routes[r][i], routes[r][i + 1])] += ods[r]
        return flows


# ===================================== flow balance model =====================================
class FlowBalance(object):
    """
    @parameter: ods,   type: dict-{index: demands},   mean: OD pairs and its demands
    @parameter: G,     type: networkx-DiGraph,        mean: traffic network
    @parameter: M,     type: float,                   mean: the range of electric vehicles
    @parameter: cost,  type: dict-{n: cost},          mean: the cost of charging station n
    @parameter: alpha, type: float,                   mean: the parameter of BRP
    @parameter: beta,  type: float,                   mean: the parameter of BRP
    """
    def __init__(self, G, ods, cost, M, pot_nodes=None, alpha=0.15, beta=1, eps=1e-4, links=None):
        self.network = copy.deepcopy(G)     # original network
        self.M = M                          # vehicle range
        self.ods = ods                      # OD pairs and its demand size
        self.R = list(ods.keys())           # OD pairs
        self.cost = copy.deepcopy(cost)     # the cost of each node
        self.pot_nodes = pot_nodes          # potential charging station locations (default: all nodes of network)
        if pot_nodes is None:
            self.pot_nodes = list(self.network.nodes())
        self.alpha = alpha                  # the parameter of BRP
        self.beta = beta                    # the parameter of BRP
        self.eps = eps                      # the error control
        # 辅助变量
        self.links = copy.deepcopy(links)
        # preprocess
        self.all_shortest_paths = dict(nx.all_pairs_dijkstra(self.network, cutoff=self.M, weight='d'))
        self.mapping = {}                   # mapping: edge of extended network to path of original network
        self.inverse = {}                   # mapping: edge of original network to edge list of extended network
        self.__var_x = []                   # variables of extended network's edges
        self.G = nx.MultiDiGraph()          # extended network
        self.G.add_nodes_from(self.network.nodes())
        self.__preprogress__()
        # Non-standard results
        self.__virtual_flow = {}
        # store results
        self.__stations = []
        self.__flows = {}
        
    
    # initial result
    def __init_results__(self):
        self.__stations = []
        self.__flows = {}
        for e in self.network.edges():
            self.__flows[e] = 0
        self.__virtual_flow = {}
        for r in self.R:
            self.__virtual_flow[r] = {}

    """ get variables from virtual edges
    @parameter: public_edges, type: dict,              mean: these edges belong to every demand r.
    @parameter: demnad_edges, type: dict,              mean: these edges belong to one demand.
    @return: variable list
    """
    def __get_variables__(self, public_edges: dict, demand_edges: dict) -> list:
        var_list = []
        for n, v in public_edges.keys():
            if v not in self.pot_nodes:
                continue
            for i in self.mapping[(n, v)].keys():
                for r1, r2 in self.R:
                    var_list.append((n, v, i, r1, r2))
        for r1, r2 in self.R:
            O = "O_" + str((r1, r2))
            D = "D_" + str((r1, r2))
            for n, v in demand_edges.keys():
                if (n == O and v == D) or (n == O and v in self.pot_nodes) or (v == D):
                    for i in self.mapping[(n, v)].keys():
                        var_list.append((n, v, i, r1, r2))
        return var_list
    
    # calc path length in the network
    def __get_path_length__(self, path):
        length = 0
        for i in range(len(path) - 1):
            length += self.network.edges[(path[i], path[i+1])]['d']
        return length

    """ get all equivalent edges from the DiGraph.
    @parameter: mapping, type: dict, mean: edge of extended network to path of original network
    @return: mapping, type dict
    """
    def __find_equivalent_edges__(self, mapping):
        new_mapping = {}
        for n, v in mapping.keys():
            length = nx.shortest_path_length(self.network, mapping[(n, v)][0], mapping[(n, v)][-1], weight='d')
            new_mapping[(n, v)] = {}
            # equivalent shortest path
            for p in nx.all_shortest_paths(self.network, mapping[(n, v)][0], mapping[(n, v)][-1], weight='d'):
                new_mapping[(n, v)][len(new_mapping[(n, v)])] = p
            # randomly identify the 'second' shortest path
            for p in nx.all_shortest_paths(self.network, mapping[(n, v)][0], mapping[(n, v)][-1]):
                if self.__get_path_length__(p) != length and self.__get_path_length__(p) <= self.M:
                    new_mapping[(n, v)][len(new_mapping[(n,v)])] = p
        return new_mapping
    
    # get mapping: edge of original network to edge list of extended network
    def __inverse_mapping__(self) -> dict:
        inverse = {}
        # inital
        for e in self.network.edges():
            inverse[e] = []
        # interation
        for u, v in self.mapping.keys():
            for i in self.mapping[(u, v)]:
                for j in range(len(self.mapping[(u, v)][i]) - 1):
                    inverse[(self.mapping[(u, v)][i][j], self.mapping[(u, v)][i][j+1])].append((u, v, i))
        return inverse
    
    # tool: add virtual edges to MultiDiGraph G
    def __add_virtual_edges__(self, edges):
        for n, v in edges.keys():
            for i in self.mapping[(n, v)].keys():
                self.G.add_edge(n, v, d=edges[(n, v)])
    
    # find: edges not utilized any path
    def __find_bad_edges__(self):
        edge_list = {}
        for e in self.network.edges():
            if self.all_shortest_paths[e[0]][0][e[1]] != self.network.edges[e]['d']:
                edge_list[e] = self.all_shortest_paths[e[0]][0][e[1]]
        return edge_list
    
    # reduce: edges not utilized any path - get the 'second' shortest path
    def __get_edges_add__(self, edge_list):
        new_mapping  = {}
        for k in self.mapping.keys():
            new_mapping[k] = {}
        for e in edge_list.keys():
            for k in self.mapping.keys():
                for i in self.mapping[k].keys():
                    idx1 = np.argwhere(np.array(self.mapping[k][i]) == e[0])
                    idx2 = np.argwhere(np.array(self.mapping[k][i]) == e[1])
                    idx1 = -1 if len(idx1) == 0 else idx1[0][0]
                    idx2 = -1 if len(idx2) == 0 else idx2[0][0]
                    if idx1 != -1 and idx2 != -1 and idx1 < idx2:
                        l = self.all_shortest_paths[self.mapping[k][i][0]][0][self.mapping[k][i][-1]] - edge_list[e] + self.network.edges[e]['d']
                        if l <= self.M:
                            new_mapping[k][len(new_mapping[k].keys())] = self.mapping[k][i][: idx1+1] + self.mapping[k][i][idx2:]
        for k in self.mapping.keys():
            for cnt in new_mapping[k].keys():
                self.mapping[k][len(self.mapping[k])] = new_mapping[k][cnt]
    
    def __get_variables_by_links__(self) -> list:
        var_list = []
        for n, v in self.links.keys():
            if v not in self.pot_nodes:
                continue
            for i in self.mapping[(n, v)].keys():
                for r1, r2 in self.R:
                    var_list.append((n, v, i, r1, r2))
        for r1, r2 in self.R:
            O = "O_" + str((r1, r2))
            D = "D_" + str((r1, r2))
            for n, v in self.mapping.keys():
                if (n == O and v == D) or (n == O and v in self.pot_nodes) or (v == D):
                    for i in self.mapping[(n, v)].keys():
                        var_list.append((n, v, i, r1, r2))
        return var_list

    def __preprogress__(self):
        # 1. add virtual nodes
        __add_virtual_nodes__(self.G, self.R)
        if self.links is None:
            edge_add = self.__find_bad_edges__()
            public_edges, public_mapping = __get_edges_public__(self.network, self.M)
            demand_edges, demand_mapping = __get_edges_demand__(self.network, self.R, self.M)
            self.mapping.update(self.__find_equivalent_edges__(public_mapping))
            self.mapping.update(self.__find_equivalent_edges__(demand_mapping))
            self.__get_edges_add__(edge_add)
            self.__var_x = self.__get_variables__(public_edges, demand_edges)
            self.__add_virtual_edges__(public_edges)
            self.__add_virtual_edges__(demand_edges)
            self.inverse = self.__inverse_mapping__()
        else:
            # 2. initial mapping
            for n, v in self.links.keys():
                self.mapping[(n, v)] = {}
                for r1, r2 in self.ods.keys():
                    O = 'O_' + str((r1, r2))
                    D = 'D_' + str((r1, r2))
                    self.mapping[(O, r1)] = {0: [r1]}
                    self.mapping[(r2, D)] = {0: [r2]}
                    if r1 == n and r2 == v:
                        self.mapping[(O, D)] = {}
                    if r1 == n:
                        self.mapping[(O, v)] = {}
                    if r2 == v:
                        self.mapping[(n, D)] = {}
            # 3. mapping
            for n, v in self.links.keys():
                idx = 0
                for p in self.links[(n, v)]:
                    l = 0
                    for i in range(len(p) - 1):
                        l += (self.network.edges[(p[i], p[i+1])]['d'])
                    self.mapping[(n, v)][idx] = list(p)
                    self.G.add_edge(n, v, d=l)
                    for r1, r2 in self.ods.keys():
                        O = 'O_' + str((r1, r2))
                        D = 'D_' + str((r1, r2))
                        if r1 == n and r2 == v:
                            self.mapping[(O, D)][idx] = list(p)
                            self.G.add_edge(O, D, d=l)
                        if r1 == n:
                            self.mapping[(O, v)][idx] = list(p)
                            self.G.add_edge(O, v, d=l)
                        if r2 == v:
                            self.mapping[(n, D)][idx] = list(p)
                            self.G.add_edge(n, D, d=l)
                    idx += 1
            # 4. inverse
            self.inverse = self.__inverse_mapping__()
            # 5. variables
            self.__var_x = self.__get_variables_by_links__()


    def INT_BPR(self, f, FFT, C):
        return FFT * (f + (self.alpha * C / (self.beta + 1)) * ((1.0 * f / C) ** (self.beta + 1)))
    
    def BPR(self, f, FFT, C):
        return FFT * (1 + self.alpha * ((1.0 * f / C) ** self.beta))
    
    # get results: stations
    def get_stations(self):
        return self.__stations
    
    # get results: flows
    def get_flows(self):
        return self.__flows

    # get results: arrival rate of every station
    def get_arrival_rates(self):
        rate = {}
        for n in self.network.nodes():
            rate[n] = 0
        for od in self.__virtual_flow.keys():
            for u, v, cnt in self.__virtual_flow[od].keys():
                if v in rate.keys():
                    rate[v] += self.__virtual_flow[od][(u, v, cnt)]
        return rate

    # get results: detour cost
    def get_detour_cost(self):
        cost = {}
        for od in self.ods.keys():
            cost[str(od)] = 0
            for u, v, cnt in self.__virtual_flow[od].keys():
                cost[str(od)] += (self.G.edges[(u, v, cnt)]['d'] * self.__virtual_flow[od][(u, v, cnt)])
            cost[str(od)] /= self.ods[od]
        return cost
    
    # get results: multiple paths
    def get_multiple_paths(self):
        od_paths = {}
        for r in self.R:
            graph = nx.DiGraph()
            for u, v, k in self.__virtual_flow[r].keys():
                path = self.mapping[(u, v)][k]
                for n in path:
                    graph.add_node(n, pos=self.network.nodes[n]['pos'])
                graph.add_nodes_from(path)
                for i in range(len(path) - 1):
                    f = self.__virtual_flow[r][(u, v, k)]
                    if (path[i], path[i+1]) in graph.edges():
                        f = graph.edges[(path[i], path[i+1])]['f'] + self.__virtual_flow[r][(u, v, k)]
                    graph.add_edge(path[i], path[i+1], d=self.network.edges[(path[i], path[i+1])]['d'], f=f)
            od_paths[r] = graph
        return od_paths

    # linearlize the objective function
    def objective_linear(self, x, x0):
        expr = LinExpr()
        i = 0
        for e in self.network.edges():
            expr += (x[e] * self.BPR(x0[i], self.network.edges[e]['FFT'], self.network.edges[e]['C']))
            i += 1
        return expr

    # actual objective function
    def objective_step(self, x):
        expr = 0
        i = 0
        for _, _, data in self.network.edges(data=True):
            expr += (self.INT_BPR(x[i], data["FFT"], data["C"]))
            i += 1
        return expr

    # search the gradient of point x
    def gradient_step(self, x):
        expr = []
        i = 0
        for _, _, data in self.network.edges(data=True):
            expr.append(self.BPR(x[i], data["FFT"], data["C"]))
            i += 1
        return expr
    
    """ Problem (18): Under construction cost limited
    @parameters: p the limit of construction cost
    @parameters: z0 the initial point of variable z (the flow of each edge)
    @return: optimal travel cost
    """
    def opt_FW(self, p, z0, eps=1e-4, max_iter=3000):
        self.__init_results__()
        x0 = np.array([0] * len(self.__var_x))
        y0 = np.array([1] * len(self.pot_nodes))
        for j in range(max_iter):
            lin_m = Model()
            lin_m.setParam('OutputFlag', 0)
            x = lin_m.addVars(self.__var_x, vtype=GRB.CONTINUOUS, name='z')
            y = lin_m.addVars(self.pot_nodes, vtype=GRB.BINARY, name='y')
            z = lin_m.addVars(self.network.edges(), vtype=GRB.CONTINUOUS, name='x')
            lin_m.setObjective(self.objective_linear(z, z0), GRB.MINIMIZE)
            lin_m.addConstrs((x.sum('O_' + str((r1, r2)), '*', '*', r1, r2) - x.sum('*', 'O_' + str((r1, r2)), '*', r1, r2) == self.ods[(r1, r2)] for r1, r2 in self.R), 'origin')
            lin_m.addConstrs((x.sum('D_' + str((r1, r2)), '*', '*', r1, r2) - x.sum('*', 'D_' + str((r1, r2)), '*', r1, r2) == -self.ods[(r1, r2)] for r1, r2 in self.R), 'destination')
            for i in self.network.nodes():
                lin_m.addConstrs((x.sum(i, '*', '*', r1, r2) - x.sum('*', i, '*', r1, r2) == 0 for r1, r2 in self.R), 'flow-balance')
            for i in self.pot_nodes:
                lin_m.addConstrs((x.sum('*', i, '*', r1, r2) / self.ods[(r1, r2)] <= y[i] for r1, r2 in self.R), 'station')
            for e in self.network.edges():
                expr = LinExpr()
                for p1, p2, cnt in self.inverse[e]:
                    expr += x.sum(p1, p2, cnt, '*', '*')
                lin_m.addConstr((z[e] == expr), 'flow-calc')
            lin_m.addConstr(quicksum(y[i] * self.cost[i] for i in self.pot_nodes) <= p, 'station-limits')
            lin_m.update()
            lin_m.optimize()
            if lin_m.status != GRB.Status.OPTIMAL:
                print('Optimal solution not found!')
                return None
            zt = np.array(lin_m.getAttr('X', z.values()))
            # find the KKT point
            if math.fabs(np.dot(zt-z0, self.gradient_step(z0))) < 1e-4:
                break
            xt = np.array(lin_m.getAttr('X', x.values()))
            yt = np.array(lin_m.getAttr('X', y.values()))
            # line search: get the length of step
            step_m = line_search(f=self.objective_step, myfprime=self.gradient_step, xk=z0, pk=zt-z0, amax=1)
            step = step_m[0] if step_m[0] is not None else 1.0
            z1 = z0 + step * (zt - z0)
            err = np.linalg.norm(z1 - z0, ord=2) / np.sum(z0)
            # update parameters
            z0 = z1
            x0 = x0 + step * (xt - x0)
            y0 = yt
            if err < eps:
                break
        # results:
        for i in range(len(y0)):
            if y0[i] == 1:
                self.__stations.append(self.pot_nodes[i])
        for i in range(len(self.__var_x)):
            if x0[i] > self.eps:
                tuple5 = self.__var_x[i]
                self.__virtual_flow[(tuple5[-2], tuple5[-1])][(tuple5[0], tuple5[1], tuple5[2])] = x0[i]
        for i in range(len(self.network.edges())):
            if z0[i] > self.eps:
                e = list(self.network.edges())[i]
                self.__flows[e] = z0[i]
        return self.objective_step(z0)
    
    """ find the best establish scheme
    @parameter: l - the lower cost
    @parameter: r - the upper cost
    @parameter: z - the aim of travel cost
    @parameter: delta - traffic congestion tolerance
    @parameter: eps - error tolerance
    @return: (M, obj) (suggest construction cost, travel cost)
    """
    def two_phase_binary_search(self, z0, l, r, z, delta, eps, max_iter=3000):
        p = 0
        for k in range(max_iter):
            M = (r + l) / 2
            obj = self.opt_FW(M, z0)
            p_k = sum(self.__stations)
            if p == p_k:
                break
            if obj > z + delta:
                r = M
            elif obj < z - delta:
                l = M
            else:
                r = M
                break
            p = p_k
        lb = -GRB.INFINITY
        ub = M
        r = M
        while (ub - lb) > eps:
            M = (r + l) / 2
            obj = self.opt_FW(M, z0)
            if obj is not None and z + delta > obj > z - delta:
                r = M
                ub = M
            else:
                l = M
                lb = M
        return M, obj
    
    # =================================== shortest path assumption ===================================
    def objective_shortest_path(self, y):
        expr = LinExpr()
        for v in self.network.nodes():
            expr += (y[v] * self.cost[v])
        return expr

    def shortest_path_constraint(self, x):
        expr = LinExpr()
        for t in self.__var_x:
            expr += (x[t] / self.ods[(t[-2], t[-1])] * self.G.edges[(t[0], t[1], t[2])]['d'])
        return expr

    def __all_shortest_path_length(self):
        length = 0
        for o, d in self.ods.keys():
            length += nx.shortest_path_length(self.network, o, d, weight='d')
        return length
    
    # shortest path model
    def opt_shortest_path(self):
        self.__init_results__()
        m = Model()
        m.setParam('OutputFlag', 0)
        x = m.addVars(self.__var_x, vtype=GRB.CONTINUOUS, name='x')
        y = m.addVars(self.pot_nodes, vtype=GRB.BINARY, name='y')
        m.setObjective(self.objective_shortest_path(y), GRB.MINIMIZE)
        m.addConstrs((x.sum('O_' + str((r1, r2)), '*', '*', r1, r2) - x.sum('*', 'O_' + str((r1, r2)), '*', r1, r2) == self.ods[(r1, r2)] for r1, r2 in self.R), 'origin')
        m.addConstrs((x.sum('D_' + str((r1, r2)), '*', '*', r1, r2) - x.sum('*', 'D_' + str((r1, r2)), '*', r1, r2) == -self.ods[(r1, r2)] for r1, r2 in self.R), 'destination')
        for i in self.network.nodes():
            m.addConstrs((x.sum(i, '*', '*', r1, r2) - x.sum('*', i, '*', r1, r2) == 0 for r1, r2 in self.R), 'flow-balance')
        for i in self.pot_nodes:
            m.addConstrs((x.sum('*', i, '*', r1, r2) / self.ods[(r1, r2)] <= y[i] for r1, r2 in self.R), 'station')
        # shortest path constraint
        m.addConstr(self.shortest_path_constraint(x) <= self.__all_shortest_path_length())
        m.update()
        m.optimize()
        if m.Status != GRB.Status.OPTIMAL:
            print("No Solution!")
            return None
        # results:
        for i in self.pot_nodes:
            if y[i].X == 1:
                self.__stations.append(i)
        for i1, i2, i3, i4, i5 in self.__var_x:
            if x[i1, i2, i3, i4, i5].X > self.eps:
                self.__virtual_flow[(i4, i5)][(i1, i2, i3)] = x[i1, i2, i3, i4, i5].X
        return None
    
    # get the total length of shortest path of all demands
    def __sum_shortest_path__(self):
        total = 0.0
        paths = dict(nx.all_pairs_dijkstra_path_length(self.network, weight='d'))
        for r1, r2 in self.R:
            total += paths[r1][r2]
        return total
    
    # shortest path and minimun travel cost
    def shortest_path_constraint(self, x):
        expr = LinExpr()
        for t in self.__var_x:
            expr += (x[t] * self.G.edges[(t[0], t[1], t[2])]['d'] / self.ods[(t[-2], t[-1])])
        return expr
    
    # shortest path and optimize travel cost
    def opt_sp_travel_cost(self, p, z0, eps=1e-4, max_iter=3000):
        self.__init_results__()
        x0 = np.array([0] * len(self.__var_x))
        y0 = np.array([1] * len(self.pot_nodes))
        for j in range(max_iter):
            lin_m = Model()
            lin_m.setParam('OutputFlag', 0)
            x = lin_m.addVars(self.__var_x, vtype=GRB.CONTINUOUS, name='x')
            y = lin_m.addVars(self.pot_nodes, vtype=GRB.BINARY, name='y')
            z = lin_m.addVars(self.network.edges(), vtype=GRB.CONTINUOUS, name='z')
            lin_m.setObjective(self.objective_linear(z, z0), GRB.MINIMIZE)
            lin_m.addConstrs((x.sum('O_' + str((r1, r2)), '*', '*', r1, r2) - x.sum('*', 'O_' + str((r1, r2)), '*', r1, r2) == self.ods[(r1, r2)] for r1, r2 in self.R), 'origin')
            lin_m.addConstrs((x.sum('D_' + str((r1, r2)), '*', '*', r1, r2) - x.sum('*', 'D_' + str((r1, r2)), '*', r1, r2) == -self.ods[(r1, r2)] for r1, r2 in self.R), 'destination')
            for i in self.network.nodes():
                lin_m.addConstrs((x.sum(i, '*', '*', r1, r2) - x.sum('*', i, '*', r1, r2) == 0 for r1, r2 in self.R), 'flow-balance')
            for i in self.pot_nodes:
                lin_m.addConstrs((x.sum('*', i, '*', r1, r2) / self.ods[(r1, r2)] <= y[i] for r1, r2 in self.R), 'station')
            for e in self.network.edges():
                expr = LinExpr()
                for p1, p2, cnt in self.inverse[e]:
                    expr += x.sum(p1, p2, cnt, '*')
                lin_m.addConstr((expr - z[e] == 0), 'flow-calc')
            lin_m.addConstr(quicksum(y[i] * self.cost[i] for i in self.pot_nodes) <= p, 'station-limits')
            lin_m.addConstr(self.shortest_path_constraint(x) == self.__sum_shortest_path__())
            lin_m.update()
            lin_m.optimize()
            if lin_m.status != GRB.Status.OPTIMAL:
                print('Optimal solution not found!')
                return None
            zt = np.array(lin_m.getAttr('X', z.values()))
            # find the KKT point
            if math.fabs(np.dot(zt-z0, self.gradient_step(z0))) < 1e-4:
                break
            xt = np.array(lin_m.getAttr('X', x.values()))
            yt = np.array(lin_m.getAttr('X', y.values()))
            # line search: get the length of step
            step_m = line_search(f=self.objective_step, myfprime=self.gradient_step, xk=z0, pk=zt-z0, amax=1)
            step = step_m[0] if step_m[0] is not None else 1.0
            z1 = z0 + step * (zt - z0)
            err = np.linalg.norm(z1 - z0, ord=2) / np.sum(z0)
            # update parameters
            z0 = z1
            x0 = x0 + step * (xt - x0)
            y0 = yt
            if err < eps:
                break
        # results:
        for i in range(len(y0)):
            if y0[i] == 1:
                self.__stations.append(self.pot_nodes[i])
        for i in range(len(self.__var_x)):
            if x0[i] > self.eps:
                tuple5 = self.__var_x[i]
                self.__virtual_flow[(tuple5[-2], tuple5[-1])][(tuple5[0], tuple5[1], tuple5[2])] = x0[i]
        for i in range(len(self.network.edges())):
            if z0[i] > self.eps:
                e = list(self.network.edges())[i]
                self.__flows[e]= z0[i]
        return self.objective_step(z0)


    # ========================= compare ========================================
    def NABLA(self, f, FFT, C):
        return FFT * (1 + (self.alpha * (self.beta + 1)) * ((f / C) ** self.beta))
    
    def objective_linear2(self, x, x0):
        expr = LinExpr()
        i = 0
        for e in self.network.edges():
            expr += (x[e] * self.NABLA(x0[i], self.network.edges[e]['FFT'], self.network.edges[e]['C']))
            i += 1
        return expr
    
    def COST(self, f, FFT, C):
        return FFT * (f + self.alpha * ((f ** (self.beta + 1)) / (C ** self.beta)))
    
    def objective_step2(self, x):
        expr = 0
        i = 0
        for _, _, data in self.network.edges(data=True):
            expr += (self.COST(x[i], data["FFT"], data["C"]))
            i += 1
        return expr
    
    def gradient_step2(self, x):
        expr = []
        i = 0
        for _, _, data in self.network.edges(data=True):
            expr.append(self.NABLA(x[i], data["FFT"], data["C"]))
            i += 1
        return expr

    # fix the cost of stations
    def opt_FW_comp(self, p, z0, eps=1e-4, max_iter=3000):
        self.__init_results__()
        x0 = np.array([0] * len(self.__var_x))
        y0 = np.array([1] * len(self.pot_nodes))
        for j in range(max_iter):
            lin_m = Model()
            lin_m.setParam('OutputFlag', 0)
            y = lin_m.addVars(self.pot_nodes, vtype=GRB.BINARY, name='y')
            x = lin_m.addVars(self.__var_x, vtype=GRB.CONTINUOUS, name='x')
            z = lin_m.addVars(self.network.edges(), vtype=GRB.CONTINUOUS, name='z')
            lin_m.setObjective(self.objective_linear2(z, z0), GRB.MINIMIZE)
            lin_m.addConstrs((x.sum('O_' + str((r1, r2)), '*', '*', r1, r2) - x.sum('*', 'O_' + str((r1, r2)), '*', r1, r2) == self.ods[(r1, r2)] for r1, r2 in self.R), 'origin')
            lin_m.addConstrs((x.sum('D_' + str((r1, r2)), '*', '*', r1, r2) - x.sum('*', 'D_' + str((r1, r2)), '*', r1, r2) == -self.ods[(r1, r2)] for r1, r2 in self.R), 'destination')
            for i in self.network.nodes():
                lin_m.addConstrs((x.sum(i, '*', '*', r1, r2) - x.sum('*', i, '*', r1, r2) == 0 for r1, r2 in self.R), 'flow-balance')
            for i in self.pot_nodes:
                lin_m.addConstrs((x.sum('*', i, '*', r1, r2) / self.ods[(r1, r2)] <= y[i] for r1, r2 in self.R), 'station')
            for e in self.network.edges():
                expr = LinExpr()
                for p1, p2, cnt in self.inverse[e]:
                    expr += x.sum(p1, p2, cnt, '*')
                lin_m.addConstr((z[e] == expr), 'flow-calc')
            lin_m.addConstr(quicksum(y[i] * self.cost[i] for i in self.pot_nodes) <= p, 'station-limits')
            lin_m.update()
            lin_m.optimize()
            if lin_m.status != GRB.Status.OPTIMAL:
                print('Optimal solution not found!')
                return None
            zt = np.array(lin_m.getAttr('X', z.values()))
            # find the KKT point
            if math.fabs(np.dot(zt-z0, self.gradient_step2(z0))) < 1e-4:
                break
            xt = np.array(lin_m.getAttr('X', x.values()))
            yt = np.array(lin_m.getAttr('X', y.values()))
            # line search: get the length of step
            step_m = line_search(f=self.objective_step2, myfprime=self.gradient_step2, xk=z0, pk=zt-z0, amax=1)
            step = step_m[0] if step_m[0] is not None else 1.0
            z1 = z0 + step * (zt - z0)
            err = np.linalg.norm(z1 - z0, ord=2) / np.sum(z0)
            # update parameters
            z0 = z1
            x0 = x0 + step * (xt - x0)
            y0 = yt
            if err < eps:
                print("limited by eps. Iter =", j + 1)
                break
        # results:
        for i in range(len(y0)):
            if y0[i] == 1:
                self.__stations.append(self.pot_nodes[i])
        for i in range(len(self.__var_x)):
            if x0[i] > self.eps:
                tuple5 = self.__var_x[i]
                self.__virtual_flow[(tuple5[-2], tuple5[-1])][(tuple5[0], tuple5[1], tuple5[2])] = x0[i]
        for i in range(len(self.network.edges())):
            if z0[i] > self.eps:
                e = list(self.network.edges())[i]
                self.__flows[e] = z0[i]
        return self.objective_step2(z0)

