import copy
import math

import numpy as np
import networkx as nx
from gurobipy import Model, GRB, LinExpr, quicksum
from scipy.optimize import line_search



""" add virtual nodes according to one OD pair.
@parameter: __G, type: networkx(DiGraph), mean: extend traffic __network
@parameter: R, type: 2-list[(o, d)],    mean: OD pairs
@return: None
"""
def __add_virtual_nodes__(__G, R: list) -> None:
    for r in R:
        O = "O_" + str((r[0], r[1]))
        D = "D_" + str((r[0], r[1]))
        __G.add_node(O)
        __G.add_node(D)
        __G.add_edge(O, r[0], d=0)
        __G.add_edge(r[1], D, d=0)

# tool: set mapping between edge and its distance, its path of original __network
def __path_to_edge__(edges, mapping, e, d, path):
    edges[e] = d
    mapping[e] = path

""" get all path which length is less than vehicle range __veh_range
@parameter: __G, type: networkx(DiGraph), mean: original traffic __network
@parameter: R, type: 2-list[(o, d)],    mean: OD pairs
@parameter: __veh_range, type: float,             mean: vehicle range
@return: edges, mapping
"""
def __get_edges_public__(__G, __veh_range: float) -> tuple:
    public_edges = {}
    mapping = {}
    for n, (d, path) in nx.all_pairs_dijkstra(__G, cutoff=__veh_range, weight='d'):
        for v in path.keys():
            if n == v:
                continue
            __path_to_edge__(public_edges, mapping, (n, v), d[v], path[v])
    return public_edges, mapping

""" get all path of virtual node which length is less than vehicle range __veh_range
@parameter: __G, type: networkx(DiGraph), mean: original traffic __network
@parameter: R, type: 2-list[(o, d)],    mean: OD pairs
@parameter: __veh_range, type: float,             mean: vehicle range
@return: edges, mapping
"""
def __get_edges_demand__(__G, R: list, __veh_range: float) -> tuple:
    demand_edges = {}
    mapping = {}
    paths = dict(nx.all_pairs_dijkstra(__G, cutoff=__veh_range, weight='d'))
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



""" Given Path Model
[1] I. Capar, __veh_range. Kuby, V. J. Leon, and Y.-J. Tsai, 
“An arc cover–path-cover formulation and strategic analysis of alternative-fuel station locations,” 
European Journal of Operational Research, 
vol. 227, no. 1, pp. 142–151, May 2013, doi: 10.1016/j.ejor.2012.11.033.
"""
class GivenPath(object):
    def __init__(
        self,
        G: nx.DiGraph,
        ods: dict[tuple, float],
        cost: dict,
        veh_range: float,
        *,
        Paths: dict[tuple, list]=None
    ):
        """ Given Path Model
        :param ods: OD pairs and its demands
        :type ods: dict

        :param G: traffic network, attribute: 'd'-link length, 'C'-link capacity, 'FFT'-free flow time
        :type G: nx.DiGraph

        :param veh_range: the range of vehicles
        :type veh_range: float

        :param cost: the cost of station n
        :type cost: dict

        :param Paths: the paths, default=None means shortest path
        :type dict
        """
        self.__network = copy.deepcopy(G)
        self.__veh_range = veh_range
        self.__ods = copy.deepcopy(ods)
        self.__cost = copy.deepcopy(cost)
        self.__Paths = Paths
        if Paths is None:
            self.__Paths = self.__get_all_OD_shortest_paths()
        self.__A, self.__K = self.__preprocess()
        # results container
        self.__stations = []
        self.__routes = {}
    
    def __init_results(self):
        self.__stations = []
        self.__routes = {}

    # tool: get shortest path between O and D
    def __get_all_OD_shortest_paths(self):
        paths = dict(nx.all_pairs_dijkstra_path(self.__network, weight='d'))
        path_dict = {}
        for r in self.__ods.keys():
            path_dict[r] = paths[r[0]][r[1]]
        return path_dict

    # tool: get the index of the first node covered subpath's ended point
    def __get_first_index(self, path):
        d = 0
        index = 0
        for i in range(len(path) - 1):
            d += self.__network.edges[(path[i], path[i + 1])]['d']
            if d > self.__veh_range:
                index = i
                break
        return index
    
    # tool: node n in path p covered edges
    def __covered_edges(self, path, n):
        edges = []
        d = 0
        for i in range(n, len(path) - 1):
            d += self.__network.edges[(path[i], path[i+1])]['d']
            if d > self.__veh_range:
                break
            edges.append((path[i], path[i+1]))
        return edges

    def __preprocess(self):
        A = {}  # need covered subpath
        K = {}  # some arcs from the path can be covered by some nodes
        for p in self.__Paths.keys():
            path = self.__Paths[p]
            index = self.__get_first_index(path)
            if index == 0:  # from O to D directly
                A[p] = []
                continue
            A[p] = path[index: ]
            # initial K_p: edge p in path can be covered some nodes
            K[p] = {}
            for i in range(len(A[p]) - 1):
                K[p][(A[p][i], A[p][i + 1])] = []
            for i in range(len(path) - 1):
                edges = self.__covered_edges(path, i)
                for e in edges:
                    if e in K[p].keys():
                        K[p][e].append(path[i])
        return A, K

    # cover all path(/flow) minimize cost
    def opt_all_cover(self):
        self.__init_results()
        m = Model()
        m.setParam('OutputFlag', 0)
        x = m.addVars(self.__network.nodes(), vtype=GRB.BINARY, name="station")
        m.setObjective(quicksum(x[i] * self.__cost[i] for i in self.__network.nodes()), GRB.MINIMIZE)
        for r in self.__ods.keys():
            path = self.__A[r]
            for p in range(len(path) - 1):
                expr = LinExpr()
                for i in self.__K[r][(path[p], path[p + 1])]:
                    expr += x[i]
                m.addConstr(expr >= 1, name="constr" + str(r))
        m.update()
        m.optimize()
        if m.status == GRB.Status.OPTIMAL:
            for i in self.__network.nodes():
                if x[i].X == 1:
                    self.__stations.append(i)
            self.__routes = self.__Paths
            return m.ObjVal
        else:
            return None
    
    # cover max flow limited by the cost of stations
    def opt_max_cover(self, limit_cost: float) -> float:
        self.__init_results()
        m = Model()
        m.setParam('OutputFlag', 0)
        x = m.addVars(self.__network.nodes(), vtype=GRB.BINARY, name="station")
        y = m.addVars(self.__ods.keys(), vtype=GRB.BINARY, name='capture')
        m.setObjective(quicksum(self.__ods[r] * y[r] for r in self.__ods.keys()), GRB.MAXIMIZE)
        for r in self.__ods.keys():
            path = self.__A[r]
            for p in range(len(path) - 1):
                expr = LinExpr()
                for i in self.__K[r][(path[p], path[p + 1])]:
                    expr += x[i]
                m.addConstr(expr >= y[r], name="constr" + str(r))
        m.addConstr(x.sum() == limit_cost)
        m.update()
        m.optimize()
        if m.status == GRB.Status.OPTIMAL:
            for i in self.__network.nodes():
                if x[i].X == 1:
                    self.__stations.append(i)
            for r in self.__ods.keys():
                if y[r].X == 1:
                    self.__routes[r] = self.__Paths[r]
            return m.ObjVal
        return None
    
    # get results: stations
    def get_stations(self):
        return self.__stations
    
    # get results: routes
    def get_routes(self) -> dict[tuple, list]:
        """
        :return: the path of every OD
        :rtype: dict[tuple, list]
        """
        return self.__routes
    
    # get results: edge flows
    def get_link_flows(self) -> dict[tuple, float]:
        """
        :return: the flow of every edge(link)
        :rtype: dict[tuple, float]
        """
        flows = {}
        for e in self.__network.edges():
            flows[e] = 0
        for k in self.__ods.keys():
            path = self.__routes[k]
            for i in range(len(path) - 1):
                flows[(path[i], path[i + 1])] += self.__ods[k]
        return flows



""" Single Path Model
[2] S. A. MirHassani and R. Ebrazi, 
“A Flexible Reformulation of the Refueling Station Location Problem,” 
Transportation Science, 
vol. 47, no. 4, pp. 617–628, Nov. 2013, doi: 10.1287/trsc.1120.0430.
"""
class SinglePath(object):
    def __init__(
        self,
        G: nx.DiGraph,
        ods: list[tuple],
        cost: dict,
        veh_range: float
    ):
        """ Single Path Model
        :param ods: OD pairs and demands
        :type ods: dict[tuple, float]
        
        :param G: traffic network, attribute: 'd'-link length, 'C'-link capacity, 'FFT'-free flow time
        :type G: nx.DiGraph

        :param veh_range: the range of electric vehicles
        :type veh_range: float

        :param cost: the cost of charging station n
        :type cost: dict
        """
        self.__network = copy.deepcopy(G)
        self.__veh_range = veh_range
        self.__ods = copy.deepcopy(ods)
        self.__cost = copy.deepcopy(cost)
        # preprocess
        self.mapping = {}                   # mapping: edge of extended __network to path of original __network
        self.inverse = {}                   # mapping: edge of original __network to edge list of extended __network
        self.__var_x = []                   # variables of extended __network's edges
        self.__G = nx.DiGraph()               # extended __network
        G.add_nodes_from(self.__network.nodes())
        self.__preprocess()
        # Non-standard results
        self.__virtual_flow = {}
        # Standard results
        self.__stations = []
    
    def __init_results(self):
        for r in self.__ods.keys():
            self.__virtual_flow[r] = []
        self.__stations = []
    
    """ get variables from virtual edges
    @parameter: public_edges, type: dict,              mean: these edges belong to every demand r.
    @parameter: demnad_edges, type: dict,              mean: these edges belong to one demand.
    @return: variable list
    """
    def __get_variables(self, public_edges: dict, demand_edges: dict) -> list:
        var_list = []
        for n, v in public_edges.keys():
            for r1, r2 in self.__ods.keys():
                var_list.append((n, v, r1, r2))
        for r in self.__ods.keys():
            O = "O_" + str((r[0], r[1]))
            D = "D_" + str((r[0], r[1]))
            for n, v in demand_edges.keys():
                if n == O or v == D:
                    var_list.append((n, v, r[0], r[1]))
        return var_list
    
    # tool: add virtual edges to __G
    def __add_virtual_edges(self, edges):
        for n, v in edges.keys():
            self.__G.add_edge(n, v, d=edges[(n, v)])
    
    # get mapping: edge of original __network to edge list of extended __network
    def __inverse_mapping(self) -> dict:
        inverse = {}
        # inital
        for e in self.__network.edges():
            inverse[e] = []
        # interation
        for p in self.mapping.keys():
            for i in range(len(self.mapping[p]) - 1):
                inverse[(self.mapping[p][i], self.mapping[p][i+1])].append(p)
        return inverse

    def __preprocess(self):
        __add_virtual_nodes__(self.__G, self.__ods.keys())
        public_edges, public_mapping = __get_edges_public__(self.__network, self.__veh_range)
        demand_edges, demand_mapping = __get_edges_demand__(self.__network, self.__ods.keys(), self.__veh_range)
        self.__var_x = self.__get_variables(public_edges, demand_edges)
        self.__add_virtual_edges(demand_edges)
        self.__add_virtual_edges(public_edges)
        self.mapping.update(demand_mapping)
        self.mapping.update(public_mapping)
        self.inverse.update(self.__inverse_mapping__())

    """ optimal the location of cahrging station
    @parameter: num, type: int, mean: the number of station, default: None.
    @return: int, mean: the number of stations.
    """
    def opt(self, limits=None):
        self.__init_results()
        m = Model()
        m.setParam('OutputFlag', 0)
        x = m.addVars(self.__var_x, vtype=GRB.BINARY)
        y = m.addVars(self.__network.nodes(), vtype=GRB.BINARY)
        m.setObjective(quicksum(y[i] * self.__cost[i] for i in self.__network.nodes()), GRB.MINIMIZE)
        m.addConstrs((x.sum('O_' + str((r1, r2)), '*', r1, r2) - x.sum('*', 'O_' + str((r1, r2)), r1, r2) == 1 for r1, r2 in self.__ods.keys()),'origin')
        m.addConstrs((x.sum('D_' + str((r1, r2)), '*', r1, r2) - x.sum('*', 'D_' + str((r1, r2)), r1, r2) == -1 for r1, r2 in self.__ods.keys()),'destination')
        for i in self.__network.nodes():
            m.addConstrs((x.sum(i, '*', r1, r2) - x.sum('*', i, r1, r2) == 0 for r1, r2 in self.__ods.keys()), 'edges')
            m.addConstrs((x.sum('*', i, r1, r2) <= y[i] for r1, r2 in self.__ods.keys()), 'station')
        if limits is not None:
            m.addConstr(y.sum() == limits, 'limits')
        m.update()
        m.optimize()
        if m.status == GRB.Status.OPTIMAL:
            for i in self.__network.nodes():
                if y[i].X == 1:
                    self.__stations.append(i)
            for e1, e2, r1, r2 in self.__var_x:
                if x[e1, e2, r1, r2].X == 1:
                    self.__virtual_flow[(r1, r2)].append((e1, e2))
            return m.ObjVal
        return None
    
    # get results: stations
    def get_stations(self) -> list:
        return self.__stations
    
    # get results: routes
    def get_routes(self) -> dict[tuple, list]:
        """
        :return: the path of every OD
        :rtype: dict[tuple, list]
        """
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
    
    def get_link_flows(self):
        """
        :return: link flow
        :rtype: dict[tuple, float]
        """
        flows = {}
        for e in self.__network.edges():
            flows[e] = 0
        routes = self.get_routes()
        for r in routes.keys():
            for i in range(len(routes[r]) - 1):
                flows[(routes[r][i], routes[r][i + 1])] += self.__ods[r]
        return flows



""" Multiple Path Model(Our)
[3] Wait to publish.
"""
class MultiPath(object):
    def __init__(
        self,
        G: nx.DiGraph,
        ods: dict[tuple, float],
        cost: dict,
        veh_range: float,
        *,
        pot_nodes: list=None,
        alpha: float=0.15,
        beta: float=1,
        eps: float=1e-4,
        links: dict=None
    ):
        """
        :param ods: OD pairs and its demands
        :type ods: dict[tuple, float]

        :param G: traffic network, attribute: 'd'-link length, 'C'-link capacity, 'FFT'-free flow time
        :type G: nx.DiGraph

        :param veh_range: the range of vehicles
        :type veh_range: float

        :param cost: the cost of station n
        :type cost: dict

        :param alpha: the parameter of BRP
        :type alpha: float
        
        :param beta: the parameter of BRP(power)
        :type beta: float
        """
        self.__network = copy.deepcopy(G)
        self.__veh_range = veh_range
        self.__ods = ods
        self.__cost = copy.deepcopy(cost)
        self.pot_nodes = pot_nodes
        if pot_nodes is None:
            self.pot_nodes = list(self.__network.nodes())
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        # 辅助变量
        self.__links = copy.deepcopy(links)
        # preprocess
        self.all_shortest_paths = dict(nx.all_pairs_dijkstra(self.__network, cutoff=self.__veh_range, weight='d'))
        self.mapping = {}                   # mapping: edge of extended __network to path of original __network
        self.inverse = {}                   # mapping: edge of original __network to edge list of extended __network
        self.__var_x = []                   # variables of extended __network's edges
        self.__G = nx.MultiDiGraph()          # extended __network
        self.__G.add_nodes_from(self.__network.nodes())
        self.__preprogress()
        # Non-standard results
        self.__virtual_flow = {}
        # store results
        self.__stations = []
        self.__flows = {}
        
    
    # initial result
    def __init_results(self):
        self.__stations = []
        self.__flows = {}
        for e in self.__network.edges():
            self.__flows[e] = 0
        self.__virtual_flow = {}
        for r in self.__ods.keys():
            self.__virtual_flow[r] = {}

    """ get variables from virtual edges
    @parameter: public_edges, type: dict,              mean: these edges belong to every demand r.
    @parameter: demnad_edges, type: dict,              mean: these edges belong to one demand.
    @return: variable list
    """
    def __get_variables(self, public_edges: dict, demand_edges: dict) -> list:
        var_list = []
        for n, v in public_edges.keys():
            if v not in self.pot_nodes:
                continue
            for i in self.mapping[(n, v)].keys():
                for r1, r2 in self.__ods.keys():
                    var_list.append((n, v, i, r1, r2))
        for r1, r2 in self.__ods.keys():
            O = "O_" + str((r1, r2))
            D = "D_" + str((r1, r2))
            for n, v in demand_edges.keys():
                if (n == O and v == D) or (n == O and v in self.pot_nodes) or (v == D):
                    for i in self.mapping[(n, v)].keys():
                        var_list.append((n, v, i, r1, r2))
        return var_list
    
    # calc path length in the __network
    def __get_path_length(self, path):
        length = 0
        for i in range(len(path) - 1):
            length += self.__network.edges[(path[i], path[i+1])]['d']
        return length

    """ get all equivalent edges from the DiGraph.
    @parameter: mapping, type: dict, mean: edge of extended __network to path of original __network
    @return: mapping, type dict
    """
    def __find_equivalent_edges(self, mapping):
        new_mapping = {}
        for n, v in mapping.keys():
            length = nx.shortest_path_length(self.__network, mapping[(n, v)][0], mapping[(n, v)][-1], weight='d')
            new_mapping[(n, v)] = {}
            # equivalent shortest path
            for p in nx.all_shortest_paths(self.__network, mapping[(n, v)][0], mapping[(n, v)][-1], weight='d'):
                new_mapping[(n, v)][len(new_mapping[(n, v)])] = p
            # randomly identify the 'second' shortest path
            for p in nx.all_shortest_paths(self.__network, mapping[(n, v)][0], mapping[(n, v)][-1]):
                if self.__get_path_length__(p) != length and self.__get_path_length__(p) <= self.__veh_range:
                    new_mapping[(n, v)][len(new_mapping[(n,v)])] = p
        return new_mapping
    
    # get mapping: edge of original __network to edge list of extended __network
    def __inverse_mapping(self) -> dict:
        inverse = {}
        # inital
        for e in self.__network.edges():
            inverse[e] = []
        # interation
        for u, v in self.mapping.keys():
            for i in self.mapping[(u, v)]:
                for j in range(len(self.mapping[(u, v)][i]) - 1):
                    inverse[(self.mapping[(u, v)][i][j], self.mapping[(u, v)][i][j+1])].append((u, v, i))
        return inverse
    
    # tool: add virtual edges to MultiDiGraph __G
    def __add_virtual_edges(self, edges):
        for n, v in edges.keys():
            for i in self.mapping[(n, v)].keys():
                self.__G.add_edge(n, v, d=edges[(n, v)])
    
    # find: edges not utilized any path
    def __find_bad_edges(self):
        edge_list = {}
        for e in self.__network.edges():
            if self.all_shortest_paths[e[0]][0][e[1]] != self.__network.edges[e]['d']:
                edge_list[e] = self.all_shortest_paths[e[0]][0][e[1]]
        return edge_list
    
    # reduce: edges not utilized any path - get the 'second' shortest path
    def __get_edges_add(self, edge_list):
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
                        l = self.all_shortest_paths[self.mapping[k][i][0]][0][self.mapping[k][i][-1]] - edge_list[e] + self.__network.edges[e]['d']
                        if l <= self.__veh_range:
                            new_mapping[k][len(new_mapping[k].keys())] = self.mapping[k][i][: idx1+1] + self.mapping[k][i][idx2:]
        for k in self.mapping.keys():
            for cnt in new_mapping[k].keys():
                self.mapping[k][len(self.mapping[k])] = new_mapping[k][cnt]
    
    def __get_variables_by_links(self) -> list:
        var_list = []
        for n, v in self.__links.keys():
            if v not in self.pot_nodes:
                continue
            for i in self.mapping[(n, v)].keys():
                for r1, r2 in self.__ods.keys():
                    var_list.append((n, v, i, r1, r2))
        for r1, r2 in self.__ods.keys():
            O = "O_" + str((r1, r2))
            D = "D_" + str((r1, r2))
            for n, v in self.mapping.keys():
                if (n == O and v == D) or (n == O and v in self.pot_nodes) or (v == D):
                    for i in self.mapping[(n, v)].keys():
                        var_list.append((n, v, i, r1, r2))
        return var_list

    def __preprogress(self):
        # 1. add virtual nodes
        __add_virtual_nodes__(self.__G, self.__ods.keys())
        if self.__links is None:
            edge_add = self.__find_bad_edges__()
            public_edges, public_mapping = __get_edges_public__(self.__network, self.__veh_range)
            demand_edges, demand_mapping = __get_edges_demand__(self.__network, self.__ods.keys(), self.__veh_range)
            self.mapping.update(self.__find_equivalent_edges(public_mapping))
            self.mapping.update(self.__find_equivalent_edges(demand_mapping))
            self.__get_edges_add(edge_add)
            self.__var_x = self.__get_variables(public_edges, demand_edges)
            self.__add_virtual_edges(public_edges)
            self.__add_virtual_edges(demand_edges)
            self.inverse = self.__inverse_mapping()
        else:
            # 2. initial mapping
            for n, v in self.__links.keys():
                self.mapping[(n, v)] = {}
                for r1, r2 in self.__ods.keys():
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
            for n, v in self.__links.keys():
                idx = 0
                for p in self.__links[(n, v)]:
                    l = 0
                    for i in range(len(p) - 1):
                        l += (self.__network.edges[(p[i], p[i+1])]['d'])
                    self.mapping[(n, v)][idx] = list(p)
                    self.__G.add_edge(n, v, d=l)
                    for r1, r2 in self.__ods.keys():
                        O = 'O_' + str((r1, r2))
                        D = 'D_' + str((r1, r2))
                        if r1 == n and r2 == v:
                            self.mapping[(O, D)][idx] = list(p)
                            self.__G.add_edge(O, D, d=l)
                        if r1 == n:
                            self.mapping[(O, v)][idx] = list(p)
                            self.__G.add_edge(O, v, d=l)
                        if r2 == v:
                            self.mapping[(n, D)][idx] = list(p)
                            self.__G.add_edge(n, D, d=l)
                    idx += 1
            # 4. inverse
            self.inverse = self.__inverse_mapping()
            # 5. variables
            self.__var_x = self.__get_variables_by_links()


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
        for n in self.__network.nodes():
            rate[n] = 0
        for od in self.__virtual_flow.keys():
            for u, v, cnt in self.__virtual_flow[od].keys():
                if v in rate.keys():
                    rate[v] += self.__virtual_flow[od][(u, v, cnt)]
        return rate

    # get results: detour __cost
    def get_detour_cost(self):
        cost = {}
        for od in self.__ods.keys():
            cost[str(od)] = 0
            for u, v, cnt in self.__virtual_flow[od].keys():
                cost[str(od)] += (self.__G.edges[(u, v, cnt)]['d'] * self.__virtual_flow[od][(u, v, cnt)])
            cost[str(od)] /= self.__ods[od]
        return cost
    
    # get results: multiple paths
    def get_multiple_paths(self):
        od_paths = {}
        for r in self.__ods.keys():
            graph = nx.DiGraph()
            for u, v, k in self.__virtual_flow[r].keys():
                path = self.mapping[(u, v)][k]
                for n in path:
                    graph.add_node(n, pos=self.__network.nodes[n]['pos'])
                graph.add_nodes_from(path)
                for i in range(len(path) - 1):
                    f = self.__virtual_flow[r][(u, v, k)]
                    if (path[i], path[i+1]) in graph.edges():
                        f = graph.edges[(path[i], path[i+1])]['f'] + self.__virtual_flow[r][(u, v, k)]
                    graph.add_edge(path[i], path[i+1], d=self.__network.edges[(path[i], path[i+1])]['d'], f=f)
            od_paths[r] = graph
        return od_paths

    # linearlize the objective function
    def objective_linear(self, x, x0):
        expr = LinExpr()
        i = 0
        for e in self.__network.edges():
            expr += (x[e] * self.BPR(x0[i], self.__network.edges[e]['FFT'], self.__network.edges[e]['C']))
            i += 1
        return expr

    # actual objective function
    def objective_step(self, x):
        expr = 0
        i = 0
        for _, _, data in self.__network.edges(data=True):
            expr += (self.INT_BPR(x[i], data["FFT"], data["C"]))
            i += 1
        return expr

    # search the gradient of point x
    def gradient_step(self, x):
        expr = []
        i = 0
        for _, _, data in self.__network.edges(data=True):
            expr.append(self.BPR(x[i], data["FFT"], data["C"]))
            i += 1
        return expr
    
    """ Problem (18): Under construction __cost limited
    @parameters: p the limit of construction __cost
    @parameters: z0 the initial point of variable z (the flow of each edge)
    @return: optimal travel __cost
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
            z = lin_m.addVars(self.__network.edges(), vtype=GRB.CONTINUOUS, name='x')
            lin_m.setObjective(self.objective_linear(z, z0), GRB.MINIMIZE)
            lin_m.addConstrs((x.sum('O_' + str((r1, r2)), '*', '*', r1, r2) - x.sum('*', 'O_' + str((r1, r2)), '*', r1, r2) == self.__ods[(r1, r2)] for r1, r2 in self.__ods.keys()), 'origin')
            lin_m.addConstrs((x.sum('D_' + str((r1, r2)), '*', '*', r1, r2) - x.sum('*', 'D_' + str((r1, r2)), '*', r1, r2) == -self.__ods[(r1, r2)] for r1, r2 in self.__ods.keys()), 'destination')
            for i in self.__network.nodes():
                lin_m.addConstrs((x.sum(i, '*', '*', r1, r2) - x.sum('*', i, '*', r1, r2) == 0 for r1, r2 in self.__ods.keys()), 'flow-balance')
            for i in self.pot_nodes:
                lin_m.addConstrs((x.sum('*', i, '*', r1, r2) / self.__ods[(r1, r2)] <= y[i] for r1, r2 in self.__ods.keys()), 'station')
            for e in self.__network.edges():
                expr = LinExpr()
                for p1, p2, cnt in self.inverse[e]:
                    expr += x.sum(p1, p2, cnt, '*', '*')
                lin_m.addConstr((z[e] == expr), 'flow-calc')
            lin_m.addConstr(quicksum(y[i] * self.__cost[i] for i in self.pot_nodes) <= p, 'station-limits')
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
        for i in range(len(self.__network.edges())):
            if z0[i] > self.eps:
                e = list(self.__network.edges())[i]
                self.__flows[e] = z0[i]
        return self.objective_step(z0)
    
    """ find the best establish scheme
    @parameter: l - the lower __cost
    @parameter: r - the upper __cost
    @parameter: z - the aim of travel __cost
    @parameter: delta - traffic congestion tolerance
    @parameter: eps - error tolerance
    @return: (__veh_range, obj) (suggest construction __cost, travel __cost)
    """
    def two_phase_binary_search(self, z0, l, r, z, delta, eps, max_iter=3000):
        p = 0
        for k in range(max_iter):
            __veh_range = (r + l) / 2
            obj = self.opt_FW(__veh_range, z0)
            p_k = sum(self.__stations)
            if p == p_k:
                break
            if obj > z + delta:
                r = __veh_range
            elif obj < z - delta:
                l = __veh_range
            else:
                r = __veh_range
                break
            p = p_k
        lb = -GRB.INFINITY
        ub = __veh_range
        r = __veh_range
        while (ub - lb) > eps:
            __veh_range = (r + l) / 2
            obj = self.opt_FW(__veh_range, z0)
            if obj is not None and z + delta > obj > z - delta:
                r = __veh_range
                ub = __veh_range
            else:
                l = __veh_range
                lb = __veh_range
        return __veh_range, obj
    
    # =================================== shortest path assumption ===================================
    def objective_shortest_path(self, y):
        expr = LinExpr()
        for v in self.__network.nodes():
            expr += (y[v] * self.__cost[v])
        return expr

    def shortest_path_constraint(self, x):
        expr = LinExpr()
        for t in self.__var_x:
            expr += (x[t] / self.__ods[(t[-2], t[-1])] * self.__G.edges[(t[0], t[1], t[2])]['d'])
        return expr

    def __all_shortest_path_length(self):
        length = 0
        for o, d in self.__ods.keys():
            length += nx.shortest_path_length(self.__network, o, d, weight='d')
        return length
    
    # shortest path model
    def opt_shortest_path(self):
        self.__init_results__()
        m = Model()
        m.setParam('OutputFlag', 0)
        x = m.addVars(self.__var_x, vtype=GRB.CONTINUOUS, name='x')
        y = m.addVars(self.pot_nodes, vtype=GRB.BINARY, name='y')
        m.setObjective(self.objective_shortest_path(y), GRB.MINIMIZE)
        m.addConstrs((x.sum('O_' + str((r1, r2)), '*', '*', r1, r2) - x.sum('*', 'O_' + str((r1, r2)), '*', r1, r2) == self.__ods[(r1, r2)] for r1, r2 in self.__ods.keys()), 'origin')
        m.addConstrs((x.sum('D_' + str((r1, r2)), '*', '*', r1, r2) - x.sum('*', 'D_' + str((r1, r2)), '*', r1, r2) == -self.__ods[(r1, r2)] for r1, r2 in self.__ods.keys()), 'destination')
        for i in self.__network.nodes():
            m.addConstrs((x.sum(i, '*', '*', r1, r2) - x.sum('*', i, '*', r1, r2) == 0 for r1, r2 in self.__ods.keys()), 'flow-balance')
        for i in self.pot_nodes:
            m.addConstrs((x.sum('*', i, '*', r1, r2) / self.__ods[(r1, r2)] <= y[i] for r1, r2 in self.__ods.keys()), 'station')
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
    def __sum_shortest_path(self):
        total = 0.0
        paths = dict(nx.all_pairs_dijkstra_path_length(self.__network, weight='d'))
        for r1, r2 in self.__ods.keys():
            total += paths[r1][r2]
        return total
    
    # shortest path and minimun travel __cost
    def shortest_path_constraint(self, x):
        expr = LinExpr()
        for t in self.__var_x:
            expr += (x[t] * self.__G.edges[(t[0], t[1], t[2])]['d'] / self.__ods[(t[-2], t[-1])])
        return expr
    
    # shortest path and optimize travel __cost
    def opt_sp_travel_cost(self, p, z0, eps=1e-4, max_iter=3000):
        self.__init_results__()
        x0 = np.array([0] * len(self.__var_x))
        y0 = np.array([1] * len(self.pot_nodes))
        for j in range(max_iter):
            lin_m = Model()
            lin_m.setParam('OutputFlag', 0)
            x = lin_m.addVars(self.__var_x, vtype=GRB.CONTINUOUS, name='x')
            y = lin_m.addVars(self.pot_nodes, vtype=GRB.BINARY, name='y')
            z = lin_m.addVars(self.__network.edges(), vtype=GRB.CONTINUOUS, name='z')
            lin_m.setObjective(self.objective_linear(z, z0), GRB.MINIMIZE)
            lin_m.addConstrs((x.sum('O_' + str((r1, r2)), '*', '*', r1, r2) - x.sum('*', 'O_' + str((r1, r2)), '*', r1, r2) == self.__ods[(r1, r2)] for r1, r2 in self.__ods.keys()), 'origin')
            lin_m.addConstrs((x.sum('D_' + str((r1, r2)), '*', '*', r1, r2) - x.sum('*', 'D_' + str((r1, r2)), '*', r1, r2) == -self.__ods[(r1, r2)] for r1, r2 in self.__ods.keys()), 'destination')
            for i in self.__network.nodes():
                lin_m.addConstrs((x.sum(i, '*', '*', r1, r2) - x.sum('*', i, '*', r1, r2) == 0 for r1, r2 in self.__ods.keys()), 'flow-balance')
            for i in self.pot_nodes:
                lin_m.addConstrs((x.sum('*', i, '*', r1, r2) / self.__ods[(r1, r2)] <= y[i] for r1, r2 in self.__ods.keys()), 'station')
            for e in self.__network.edges():
                expr = LinExpr()
                for p1, p2, cnt in self.inverse[e]:
                    expr += x.sum(p1, p2, cnt, '*')
                lin_m.addConstr((expr - z[e] == 0), 'flow-calc')
            lin_m.addConstr(quicksum(y[i] * self.__cost[i] for i in self.pot_nodes) <= p, 'station-limits')
            lin_m.addConstr(self.shortest_path_constraint(x) == self.__sum_shortest_path())
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
        for i in range(len(self.__network.edges())):
            if z0[i] > self.eps:
                e = list(self.__network.edges())[i]
                self.__flows[e]= z0[i]
        return self.objective_step(z0)


    # ========================= compare ========================================
    def NABLA(self, f, FFT, C):
        return FFT * (1 + (self.alpha * (self.beta + 1)) * ((f / C) ** self.beta))
    
    def objective_linear2(self, x, x0):
        expr = LinExpr()
        i = 0
        for e in self.__network.edges():
            expr += (x[e] * self.NABLA(x0[i], self.__network.edges[e]['FFT'], self.__network.edges[e]['C']))
            i += 1
        return expr
    
    def COST(self, f, FFT, C):
        return FFT * (f + self.alpha * ((f ** (self.beta + 1)) / (C ** self.beta)))
    
    def objective_step2(self, x):
        expr = 0
        i = 0
        for _, _, data in self.__network.edges(data=True):
            expr += (self.COST(x[i], data["FFT"], data["C"]))
            i += 1
        return expr
    
    def gradient_step2(self, x):
        expr = []
        i = 0
        for _, _, data in self.__network.edges(data=True):
            expr.append(self.NABLA(x[i], data["FFT"], data["C"]))
            i += 1
        return expr

    # fix the __cost of stations
    def opt_FW_comp(self, p, z0, eps=1e-4, max_iter=3000):
        self.__init_results__()
        x0 = np.array([0] * len(self.__var_x))
        y0 = np.array([1] * len(self.pot_nodes))
        for j in range(max_iter):
            lin_m = Model()
            lin_m.setParam('OutputFlag', 0)
            y = lin_m.addVars(self.pot_nodes, vtype=GRB.BINARY, name='y')
            x = lin_m.addVars(self.__var_x, vtype=GRB.CONTINUOUS, name='x')
            z = lin_m.addVars(self.__network.edges(), vtype=GRB.CONTINUOUS, name='z')
            lin_m.setObjective(self.objective_linear2(z, z0), GRB.MINIMIZE)
            lin_m.addConstrs((x.sum('O_' + str((r1, r2)), '*', '*', r1, r2) - x.sum('*', 'O_' + str((r1, r2)), '*', r1, r2) == self.__ods[(r1, r2)] for r1, r2 in self.__ods.keys()), 'origin')
            lin_m.addConstrs((x.sum('D_' + str((r1, r2)), '*', '*', r1, r2) - x.sum('*', 'D_' + str((r1, r2)), '*', r1, r2) == -self.__ods[(r1, r2)] for r1, r2 in self.__ods.keys()), 'destination')
            for i in self.__network.nodes():
                lin_m.addConstrs((x.sum(i, '*', '*', r1, r2) - x.sum('*', i, '*', r1, r2) == 0 for r1, r2 in self.__ods.keys()), 'flow-balance')
            for i in self.pot_nodes:
                lin_m.addConstrs((x.sum('*', i, '*', r1, r2) / self.__ods[(r1, r2)] <= y[i] for r1, r2 in self.__ods.keys()), 'station')
            for e in self.__network.edges():
                expr = LinExpr()
                for p1, p2, cnt in self.inverse[e]:
                    expr += x.sum(p1, p2, cnt, '*')
                lin_m.addConstr((z[e] == expr), 'flow-calc')
            lin_m.addConstr(quicksum(y[i] * self.__cost[i] for i in self.pot_nodes) <= p, 'station-limits')
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
        for i in range(len(self.__network.edges())):
            if z0[i] > self.eps:
                e = list(self.__network.edges())[i]
                self.__flows[e] = z0[i]
        return self.objective_step2(z0)

