import copy
import math
from collections.abc import Callable

import networkx as nx
import numpy as np
from scipy.optimize import line_search



class LinkBased(object):
    def __init__(
        self,
        G: nx.DiGraph,
        ods: dict[tuple, float],
        *,
        LPF: Callable[[float, float, float], float]=None,
        INT_LPF: Callable[[float, float, float], float]=None,
        DER_LPF: Callable[[float, float, float], float]=None,
        ue_type: str='UE',
        alpha: float=0.15,
        beta: float=4,
        max_iter: int=30,
        R: float=None
    ) -> None:
        """ Frank-Wolfe algorithm based on the finding shortest path method

        :param G        networkx.DiGraph                        the transpoation network

        :param ods      dict[tuple, float]                      the OD demand based on __G

        :param LPF      callable[[float, float, float], float]  link performance function[FFT, Capacity, flow] -> float
        
        :param INT_LPF  callable[[float, float, float], float]  the integral of LPF[FFT, Capacity, flow] -> float
        
        :param DER_LPF  callable[[float, float, float], float]  the partial derivative function of LPF[FFT, Capacity, flow] -> float
        
        :param ue_type  str['UE', 'SO']                         the category of ue
        
        :param alpha    float                                   the parameter of default LPF(BRP)
        
        :param beta     float                                   the parameter of default LPF(BRP)

        :param max_iter int                                     the number of line search iteration
        
        :param R        float                                   Auxiliary variable for location model
        """
        # Input parameters
        self.__G = copy.deepcopy(G)
        self.__ods = ods
        self.__type = ue_type
        self.__check_input_parameters(LPF, INT_LPF, DER_LPF)
        self.__LPF = self.__BPR
        self.__INT_LPF = self.__INT_BPR
        self.__DER_LPF = self.__DER_BPR
        self.alpha = alpha
        self.beta = beta
        self.__func = self.__INT_LPF
        self.__der_func = self.__LPF
        self.line_search_iter = max_iter
        if LPF is not None:
            self.__LPF = copy.deepcopy(LPF)
        if INT_LPF is not None:
            self.__INT_LPF = copy.deepcopy(INT_LPF)
        if DER_LPF is not None:
            self.__DER_LPF = copy.deepcopy(DER_LPF)
        if ue_type == 'SO':
            self.__func = self.__INT_SO
            self.__der_func = self.__SO
        # Intermediate parameter
        self.iter_link_flow = {}
        self.__path_cost = {}
        # Result container
        self.link_flow = {}
        # Initial parameter
        self.__paths = True if len(set([t[0] for t in list(self.__ods.keys())])) >= len(self.__G.nodes()) else False
        # Auxiliary variable for location model
        self.R = R
        self.link_set = self.__init_link_set()

    def __check_input_parameters(self, LPF, INT_LPF, DER_LPF):
        # __type
        if self.__type != 'UE' and self.__type != 'SO':
            raise ValueError("Only support __type: UE, SO!")
        # function
        if (LPF is None and INT_LPF is not None) or (LPF is not None and LPF is None):
            raise ValueError("It is not allowed for exactly one of LPF and INT_LPF to be None!")
        if self.__type == 'SO' and (DER_LPF is None and (LPF is not None or INT_LPF is not None)):
            raise ValueError("Custom LPF but not given DER_LPF!")
        # networkx
        if nx.is_empty(self.__G):
            raise ValueError("The transportation network is empty!")
        for _, _, data in self.__G.edges(data=True):
            if 'FFT' not in data.keys() or 'd' not in data.keys() or 'C' not in data.keys():
                raise ValueError("Missing properties: 'FFT', 'C' or 'd'!")
        # ods
        for o, d in self.__ods.keys():
            if o not in self.__G.nodes() or d not in self.__G.nodes():
                raise ValueError("OD data not suitable for the transportation network!")

    def __BPR(self, FFT, C, flow) -> float:
        return FFT * (1 + self.alpha * (flow / C) ** self.beta)

    def __INT_BPR(self, FFT, C, flow):
        return FFT * (flow + (self.alpha * C / (self.beta + 1)) * ((flow / C) ** (self.beta + 1)))

    def __DER_BPR(self, FFT, C, flow):
        return FFT * self.alpha * self.beta * (flow ** (self.beta - 1)) / (C ** self.beta)

    def __INT_SO(self, FFT, C, flow):
        return flow * self.__LPF(FFT, C, flow)

    def __SO(self, FFT, C, flow):
        return self.__LPF(FFT, C, flow) + flow * self.__DER_LPF(FFT, C, flow)

    def __init_link_flow(self):
        for e in self.__G.edges():
            self.link_flow[e] = 0

    def __init_iter_link_flow(self):
        for e in self.__G.edges():
            self.iter_link_flow[e] = 0

    # =================================== Auxiliary variable for location model ===================================
    def __init_link_set(self):
        if self.R is None:
            return
        link_set = {}
        for n in self.__G.nodes():
            for v in self.__G.nodes():
                if n == v:
                    continue
                if nx.shortest_path_length(self.__G, n, v, weight='d') <= self.R:
                    link_set[(n, v)] = set()
                    for p in nx.all_shortest_paths(self.__G, n, v, weight='d'):
                        link_set[(n, v)].add(tuple(p))
        return link_set

    def __update_link_set(self):
        if self.R is None:
            return
        for n in self.__G.nodes():
            for v in self.__G.nodes():
                if n == v or (n, v) not in self.link_set.keys():
                    continue
                # equivalent path
                for p in nx.all_shortest_paths(self.__G, n, v, weight='weight'):
                    l = 0
                    for i in range(len(p) - 1):
                        l += (self.__G.edges[(p[i], p[i+1])]['d'])
                    if l <= self.R:
                        self.link_set[(n, v)].add(tuple(p))
    # =================================== Auxiliary variable for location model ===================================

    # find all shortest paths between all OD pairs
    def __all_pairs_shortest_paths(self):
        if self.__paths:
            paths = nx.all_pairs_dijkstra_path(self.__G, weight="weight")
            return dict(paths)
        else:
            paths = {}
            for r in self.__ods.keys():
                if r[0] not in paths.keys():
                    paths[r[0]] = nx.single_source_dijkstra_path(self.__G, r[0], weight="weight")
            return paths

    def __set_link_flow_by_shortest_path(self):
        self.__init_iter_link_flow()
        paths = self.__all_pairs_shortest_paths()
        self.__update_link_set()
        for r in self.__ods.keys():
            path = paths[r[0]][r[1]]
            path_length = 0
            for i in range(len(path) - 1):
                self.iter_link_flow[(path[i], path[i + 1])] += self.__ods[r]
                path_length += self.__G.edges[(path[i], path[i + 1])]['weight']
            self.__path_cost[r] = path_length

    def __objective_function(self):
        s = 0
        for u, v, data in self.__G.edges(data=True):
            s += self.__func(data["FFT"], data["C"], self.link_flow[(u, v)])
        return s
    
    def __derivative_function(self, step):
        s = 0
        for u, v, data in self.__G.edges(data=True):
            e = (u, v)
            flow = self.link_flow[e] + step * (self.iter_link_flow[e] - self.link_flow[e])
            s += (self.__der_func(data["FFT"], data["C"], flow) * (self.iter_link_flow[e] - self.link_flow[e]))
        return s

    def __line_search(self):
        step_low = 0.0
        step_high = 1.0
        low = self.__derivative_function(step_low)
        high = self.__derivative_function(step_high)
        if low * high > 0:
            return 1.0
        for _ in range(self.line_search_iter):
            step_mid = (step_low + step_high) / 2
            mid = self.__derivative_function(step_mid)
            if mid < 0:
                step_low = step_mid
            elif mid == 0:
                break
            else:
                step_high = step_mid
        return (step_low + step_high) / 2

    def __update_link_weight(self):
        for u, v, data in self.__G.edges(data=True):
            self.__G.edges[(u, v)]['weight'] = self.__der_func(data["FFT"], data["C"], self.link_flow[(u, v)])

    def __update_step(self, step):
        for e in self.__G.edges():
            self.link_flow[e] += (step * (self.iter_link_flow[e] - self.link_flow[e]))

    def compute_relative_gap(self):
        A = 0
        B = 0
        for u, v, data in self.__G.edges(data=True):
            A += (self.link_flow[(u, v)] * self.__LPF(data["FFT"], data["C"], self.link_flow[(u, v)]))
        for od in self.__ods.keys():
            B += (self.__ods[od] * self.__path_cost[od])
        return (A - B) / B

    def norm_gap(self):
        A = 0
        B = 0
        for e in self.__G.edges():
            A += ((self.iter_link_flow[e] - self.link_flow[e]) ** 2)
            B += (self.link_flow[e] ** 2)
        return A / B

    def compute_gap(self):
        if self.__type == 'UE':
            return self.compute_relative_gap()
        elif self.__type == 'SO':
            return self.norm_gap()

    def opt(self, eps=3.9e-15, max_iter=20000):
        self.__init_link_flow()
        self.__update_link_weight()
        for i in range(max_iter):
            self.__set_link_flow_by_shortest_path()
            step = self.__line_search()
            self.__update_step(step)
            self.__update_link_weight()
            if self.compute_gap() < eps and i != 0:
                break
        return self.__objective_function()

    def get_system_time_cost(self):
        time = 0
        for n, v, data in self.__G.edges(data=True):
            time += (self.link_flow[(n, v)] * self.__LPF(data['FFT'], data['C'], self.link_flow[(n, v)]))
        return time

class PathBased(object):
    def __init__(
        self,
        G: nx.DiGraph,
        ods: dict[tuple, float],
        Paths: dict[tuple, list[list]],
        *,
        LPF: Callable[[float, float, float], float]=None,
        INT_LPF: Callable[[float, float, float], float]=None,
        DER_LPF: Callable[[float, float, float], float]=None,
        ue_type: str='UE',
        alpha: float=0.15,
        beta: float=4,
        theta: float=1
    ) -> None:
        """ Frank-Wolfe algorithm based on the finding shortest path method

        :param G        networkx.DiGraph                        the transpoation network

        :param ods      dict[tuple, float]                      the OD demand based on __G

        :param Paths    dict[tuple, list[list]]                 the given paths of every od

        :param LPF      callable[[float, float, float], float]  link performance function[FFT, Capacity, flow] -> float
        
        :param INT_LPF  callable[[float, float, float], float]  the integral of LPF[FFT, Capacity, flow] -> float
        
        :param DER_LPF  callable[[float, float, float], float]  the partial derivative function of LPF[FFT, Capacity, flow] -> float
        
        :param ue_type  str['UE', 'SO', 'SUE']                  the category of ue
        
        :param alpha    float                                   the parameter of default LPF(BRP)
        
        :param beta     float                                   the parameter of default LPF(BRP)

        :param theta    float                                   the parameter of SUE
        """
        # Input parameters
        self.__G = copy.deepcopy(G)
        self.__ods = ods
        self.given_paths = True
        self.Paths = Paths
        if Paths is None:
            self.given_paths = False
            self.__init_Paths()
        self.__type = ue_type
        self.__check_input_parameters(LPF, INT_LPF, DER_LPF)
        self.__LPF = self.__BPR
        self.__INT_LPF = self.__INT_BPR
        self.__DER_LPF = self.__DER_BPR
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.__func = self.__INT_LPF
        self.__der_func = self.__LPF
        self.__paths = True if len(set([t[0] for t in list(self.__ods.keys())])) >= len(self.__G.nodes()) else False
        if LPF is not None:
            self.__LPF = copy.deepcopy(LPF)
        if INT_LPF is not None:
            self.__INT_LPF = copy.deepcopy(INT_LPF)
        if DER_LPF is not None:
            self.__DER_LPF = copy.deepcopy(DER_LPF)
        if ue_type == 'SO':
            self.__func = self.__INT_SO
            self.__der_func = self.__SO
        # Intermediate parameter
        self.iter_link_flow = {}
        self.iter_path_flow = {}
        self.__path_cost = {}
        # Result container
        self.link_flow = {}
        self.path_flow = {}
        # initial parameters

    def __check_input_parameters(self, LPF, INT_LPF, DER_LPF):
        # __type
        if self.__type != 'UE' and self.__type != 'SO' and self.__type != 'SUE':
            raise ValueError("Only support __type: UE, SO, SUE!")
        # function
        if (LPF is None and INT_LPF is not None) or (LPF is not None and LPF is None):
            raise ValueError("It is not allowed for exactly one of LPF and INT_LPF to be None!")
        if self.__type == 'SO' and (DER_LPF is None and (LPF is not None or INT_LPF is not None)):
            raise ValueError("Custom LPF but not given DER_LPF!")
        if self.__type == 'SUE' and self.Paths == None:
            raise ValueError("SUE: Paths not allow to be None.")
        # networkx
        if nx.is_empty(self.__G):
            raise ValueError("The transportation network is empty!")
        for _, _, data in self.__G.edges(data=True):
            if 'FFT' not in data.keys() or 'd' not in data.keys() or 'C' not in data.keys():
                raise ValueError("Missing properties: 'FFT', 'C' or 'd'!")
        # __ods
        for o, d in self.__ods.keys():
            if o not in self.__G.nodes() or d not in self.__G.nodes():
                raise ValueError("OD data not suitable for the transportation network!")
        # Paths
        for od in self.__ods.keys():
            if od not in self.Paths:
                raise ValueError("Missing path of some od!")

    def __BPR(self, FFT, C, flow) -> float:
        return FFT * (1 + self.alpha * (flow / C) ** self.beta)

    def __INT_BPR(self, FFT, C, flow):
        return FFT * (flow + (self.alpha * C / (self.beta + 1)) * ((flow / C) ** (self.beta + 1)))

    def __DER_BPR(self, FFT, C, flow):
        return FFT * self.alpha * self.beta * (flow ** (self.beta - 1)) / (C ** self.beta)

    def __INT_SO(self, FFT, C, flow):
        return flow * self.__LPF(FFT, C, flow)

    def __SO(self, FFT, C, flow):
        return self.__LPF(FFT, C, flow) + flow * self.__DER_LPF(FFT, C, flow)

    # ============================ Initial parameters ============================
    def __init_link_flow(self):
        for e in self.__G.edges():
            self.link_flow[e] = 0

    def __init_iter_link_flow(self):
        for e in self.__G.edges():
            self.iter_link_flow[e] = 0

    def __init_path_flow(self):
        for od in self.__ods.keys():
            self.path_flow[od] = [0] * len(self.Paths[od])

    def __init_iter_path_flow(self):
        for od in self.__ods.keys():
            self.iter_path_flow[od] = [0] * len(self.Paths[od])

    def __init_Paths(self):
        self.Paths = {}
        for od in self.__ods.keys():
            self.Paths[od] = []
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Initial parameters ^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    def __get_one_path_weight(self, path):
        weight = 0
        for i in range(len(path) - 1):
            weight += (self.__G.edges[(path[i], path[i + 1])]['weight'])
        return weight

    def __get_all_path_weight(self):
        c = {}
        for od in self.Paths.keys():
            c[od] = []
            for path in self.Paths[od]:
                c[od].append(self.__get_one_path_weight(path))
        return c

    def __assignment_by_given_paths(self, cost):
        self.__init_iter_link_flow()
        self.__init_iter_path_flow()
        for od in cost.keys():
            shortest_path_length = min(cost[od])
            path_index = cost[od].index(shortest_path_length)
            self.iter_path_flow[od][path_index] = self.__ods[od]
            path = self.Paths[od][path_index]
            for i in range(len(path) - 1):
                self.iter_link_flow[(path[i], path[i + 1])] = self.__ods[od]

    # find all shortest paths between all OD pairs
    def __all_pairs_shortest_paths(self):
        if self.__paths:
            paths = nx.all_pairs_dijkstra_path(self.__G, weight="weight")
            return dict(paths)
        else:
            paths = {}
            for r in self.__ods.keys():
                if r[0] not in paths.keys():
                    paths[r[0]] = nx.single_source_dijkstra_path(self.__G, r[0], weight="weight")
            return paths
    
    def __assignment_by_dijkstra(self):
        paths = self.__all_pairs_shortest_paths()
        for o, d in self.__ods.keys():
            if paths[o][d] not in self.Paths[(o, d)]:
                self.Paths[(o, d)].append(paths[o][d])
        # Attention initial iter flow is related to Paths.
        self.__init_iter_path_flow()
        self.__init_iter_link_flow()
        for o, d in self.__ods.keys():
            path_index = self.Paths[(o, d)].index(paths[o][d])
            self.iter_path_flow[(o, d)][path_index] = self.__ods[(o, d)]
            path_length = 0
            for i in range(len(paths[o][d]) - 1):
                self.iter_link_flow[(paths[o][d][i], paths[o][d][i + 1])] += self.__ods[(o, d)]
                path_length += self.__G.edges[(paths[o][d][i], paths[o][d][i + 1])]['weight']
            self.__path_cost[(o, d)] = path_length

    def __shortest_path_assignment(self):
        if self.given_paths is True:
            cost = self.__get_all_path_weight()
            self.__assignment_by_given_paths(cost)
        else:
            self.__assignment_by_dijkstra()

    # ============================ Update parameters ============================
    def __update_link_weight(self):
        for u, v, data in self.__G.edges(data=True):
            self.__G.edges[(u, v)]['weight'] = self.__der_func(data["FFT"], data["C"], self.link_flow[(u, v)])

    def __update_by_step(self, step):
        for e in self.__G.edges():
            self.link_flow[e] += (step * (self.iter_link_flow[e] - self.link_flow[e]))
        for od in self.__ods.keys():
            for i in range(len(self.iter_path_flow[od])):
                if i >= len(self.path_flow[od]):
                    self.path_flow[od].append(step * self.iter_path_flow[od][i])
                else:
                    self.path_flow[od][i] += (step * (self.iter_path_flow[od][i] - self.path_flow[od][i]))
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^ Update parameters ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
    def __objective_function(self):
        s = 0
        for u, v, data in self.__G.edges(data=True):
            s += self.__func(data["FFT"], data["C"], self.link_flow[(u, v)])
        return s

    def compute_absolute_gap(self):
        A = 0
        B = 0
        all_cost = self.__get_all_path_weight()
        for od in self.__ods.keys():
            A += (self.__ods[od] * (max(all_cost[od]) - min(all_cost[od])))
            B += (self.__ods[od] * min(all_cost[od]))
        return A / B

    def compute_relative_gap(self):
        A = 0
        B = 0
        for u, v, data in self.__G.edges(data=True):
            A += (self.link_flow[(u, v)] * self.__LPF(data["FFT"], data["C"], self.link_flow[(u, v)]))
        for od in self.__ods.keys():
            B += (self.__ods[od] * self.__path_cost[od])
        return (A - B) / B

    def compute_gap(self):
        if self.__type == 'UE':
            return self.compute_relative_gap()
        elif self.__type == 'SO':
            return self.compute_absolute_gap()

    def __derivative_function(self, step):
        s = 0
        for u, v, data in self.__G.edges(data=True):
            e = (u, v)
            flow = self.link_flow[e] + step * (self.iter_link_flow[e] - self.link_flow[e])
            s += (self.__der_func(data["FFT"], data["C"], flow) * (self.iter_link_flow[e] - self.link_flow[e]))
        return s

    def __line_search(self):
        step_low = 0.0
        step_high = 1.0
        low = self.__derivative_function(step_low)
        high = self.__derivative_function(step_high)
        if low * high > 0:
            return 1.0
        for _ in range(30):
            step_mid = (step_low + step_high) / 2
            mid = self.__derivative_function(step_mid)
            if mid < 0:
                step_low = step_mid
            elif mid == 0:
                break
            else:
                step_high = step_mid
        return (step_low + step_high) / 2

    def opt(self, eps=3.9e-15, max_iter=20000):
        self.__init_link_flow()
        self.__init_path_flow()
        self.__update_link_weight()
        for i in range(max_iter):
            self.__shortest_path_assignment()
            step = self.__line_search()
            self.__update_by_step(step)
            self.__update_link_weight()
            if self.compute_gap() < eps and i != 0:
                break
        return self.__objective_function()

    def get_system_time_cost(self):
        time = 0
        for n, v, data in self.__G.edges(data=True):
            time += (self.link_flow[(n, v)] * self.__LPF(data['FFT'], data['C'], self.link_flow[(n, v)]))
        return time

    # SUE
    def __logit_assignment(self):
        self.__init_iter_link_flow()
        self.__init_iter_path_flow()
        all_cost = self.__get_all_path_weight()
        for od in self.__ods.keys():
            exp_utils = np.exp(-self.theta * np.array(all_cost[od]))
            probs = exp_utils / exp_utils.sum()
            k = 0
            for path, prob in zip(self.Paths[od], probs):
                self.iter_path_flow[od][k] = prob * self.__ods[od]
                for i in range(len(path) - 1):
                    self.iter_link_flow[(path[i], path[i + 1])] += self.__ods[od]
