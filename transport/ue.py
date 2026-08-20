import copy

import networkx as nx
import numpy as np
from scipy.optimize import line_search
from collections.abc import Callable



class LinkBased(object):
    def __init__(
        self,
        G: nx.DiGraph,
        ods: dict[tuple, float],
        *,
        LPF: Callable[[float, float, float], float]=None,
        INT_LPF: Callable[[float, float, float], float]=None,
        DER_LPF: Callable[[float, float, float], float]=None,
        type: str='UE',
        alpha: float=0.15,
        beta: float=4,
        R: float=None
    ) -> None:
        """ Frank-Wolfe algorithm based on the finding shortest path method

        :param G        networkx.DiGraph                        the transpoation network

        :param ods      dict[tuple, float]                      the OD demand based on G

        :param LPF      callable[[float, float, float], float]  link performance function[FFT, Capacity, flow] -> float
        
        :param INT_LPF  callable[[float, float, float], float]  the integral of LPF[FFT, Capacity, flow] -> float
        
        :param DER_LPF  callable[[float, float, float], float]  the partial derivative function of LPF[FFT, Capacity, flow] -> float
        
        :param type     str['UE', 'SO']                         the category of ue
        
        :param alpha    float                                   the parameter of default LPF(BRP)
        
        :param beta     float                                   the parameter of default LPF(BRP)
        
        :param R        float                                   Auxiliary variable for location model
        """
        # Input parameters
        self.G = copy.deepcopy(G)
        self.ods = ods
        self.type = type
        self.__check_input_parameters(LPF, INT_LPF, DER_LPF)
        self.LPF = self.__BPR
        self.INT_LPF = self.__INT_BPR
        self.DER_LPF = self.__DER_BPR
        self.alpha = alpha
        self.beta = beta
        self.func = self.INT_LPF
        self.d_func = self.LPF
        if LPF is not None:
            self.LPF = copy.deepcopy(LPF)
        if INT_LPF is not None:
            self.INT_LPF = copy.deepcopy(INT_LPF)
        if DER_LPF is not None:
            self.DER_LPF = copy.deepcopy(DER_LPF)
        if type == 'SO':
            self.func = self.__INT_SO
            self.d_func = self.__SO
        # Intermediate parameter
        self.iter_flow = {}
        # Result container
        self.link_flow = {}
        # Initial parameter
        self.__paths = True if len(set([t[0] for t in list(self.ods.keys())])) >= len(self.G.nodes()) else False
        # Auxiliary variable for location model
        self.R = R
        self.link_set = self.__init_link_set()

    def __check_input_parameters(self, LPF, INT_LPF, DER_LPF):
        # type
        if self.type != 'UE' and self.type != 'SO':
            raise ValueError("Only support type: UE, SO!")
        # function
        if (LPF is None and INT_LPF is not None) or (LPF is not None and LPF is None):
            raise ValueError("It is not allowed for exactly one of LPF and INT_LPF to be None!")
        if self.type == 'SO' and (DER_LPF is None and (LPF is not None or INT_LPF is not None)):
            raise ValueError("Custom LPF but not given DER_LPF!")
        # networkx
        if nx.is_empty(self.G):
            raise ValueError("The transportation network is empty!")
        for _, _, data in self.G.edges(data=True):
            if 'FFT' not in data.keys() or 'd' not in data.keys() or 'C' not in data.keys():
                raise ValueError("Missing properties: 'FFT', 'C' or 'd'!")
        # ods
        for o, d in self.ods.keys():
            if o not in self.G.nodes() or d not in self.G.nodes():
                raise ValueError("OD data not suitable for the transportation network!")

    def __BPR(self, FFT, C, flow) -> float:
        return FFT * (1 + self.alpha * (flow / C) ** self.beta)

    def __INT_BPR(self, FFT, C, flow):
        return FFT * (flow + (self.alpha * C / (self.beta + 1)) * ((flow / C) ** (self.beta + 1)))

    def __DER_BPR(self, FFT, C, flow):
        return FFT * self.alpha * self.beta * (flow ** (self.beta - 1)) / (C ** self.beta)

    # find all shortest paths between all OD pairs
    def __all_pairs_shortest_paths(self):
        if self.__paths:
            paths = nx.all_pairs_dijkstra_path(self.G, weight="weight")
            return dict(paths)
        else:
            paths = {}
            for r in self.ods.keys():
                if r[0] not in paths.keys():
                    paths[r[0]] = nx.single_source_dijkstra_path(self.G, r[0], weight="weight")
            return paths

    def __find_shortest_path_by_flow(self):
        self.__init_iter_flow()
        paths = self.__all_pairs_shortest_paths()
        self.__update_link_set()
        for r in self.ods.keys():
            path = paths[r[0]][r[1]]
            for i in range(len(path) - 1):
                self.iter_flow[(path[i], path[i + 1])] += self.ods[r]

    def __init_link_flow(self):
        for e in self.G.edges():
            self.link_flow[e] = 0
            self.G.edges[e]['weight'] =  self.G.edges[e]['d']
        paths = self.__all_pairs_shortest_paths()
        for r in self.ods.keys():
            path = paths[r[0]][r[1]]
            for i in range(len(path) - 1):
                self.link_flow[(path[i], path[i + 1])] += self.ods[r]
        for n1, n2, data in self.G.edges(data=True):
            data["weight"] = self.d_func(data["FFT"], data["C"], self.link_flow[(n1, n2)])

    def __init_iter_flow(self):
        for e in self.G.edges():
            self.iter_flow[e] = 0

    # =================================== Auxiliary variable for location model ===================================
    def __init_link_set(self):
        if self.R is None:
            return
        link_set = {}
        for n in self.G.nodes():
            for v in self.G.nodes():
                if n == v:
                    continue
                if nx.shortest_path_length(self.G, n, v, weight='d') <= self.R:
                    link_set[(n, v)] = set()
                    for p in nx.all_shortest_paths(self.G, n, v, weight='d'):
                        link_set[(n, v)].add(tuple(p))
        return link_set

    def __update_link_set(self):
        if self.R is None:
            return
        for n in self.G.nodes():
            for v in self.G.nodes():
                if n == v or (n, v) not in self.link_set.keys():
                    continue
                # equivalent path
                for p in nx.all_shortest_paths(self.G, n, v, weight='weight'):
                    l = 0
                    for i in range(len(p) - 1):
                        l += (self.G.edges[(p[i], p[i+1])]['d'])
                    if l <= self.R:
                        self.link_set[(n, v)].add(tuple(p))
    # =================================== Auxiliary variable for location model ===================================

    def search_step_length(self, x):
        expr = 0
        i = 0
        for _, _, data in self.G.edges(data=True):
            expr += (self.func(data["FFT"], data["C"], x[i]))
            i += 1
        return expr

    def search_step_gradient(self, x):
        expr = []
        i = 0
        for _, _, data in self.G.edges(data=True):
            expr.append(self.d_func(data["FFT"], data["C"], x[i]))
            i += 1
        return expr

    def update(self, step):
        for e in self.G.edges():
            self.link_flow[e] += (step * (self.iter_flow[e] - self.link_flow[e]))
            self.G.edges[e]['weight'] = self.d_func(self.G.edges[e]["FFT"], self.G.edges[e]["C"], self.link_flow[e])

    def ue_opt(self, eps=5e-5, max_iter=3000, norm=2):
        self.__init_link_flow()
        x0 = np.array(list(self.link_flow.values()))
        for i in range(max_iter):
            self.__find_shortest_path_by_flow()
            direction = np.array(list(self.iter_flow.values())) - x0
            step_res = line_search(self.search_step_length, self.search_step_gradient, x0, direction)
            step = step_res[0] if step_res[0] is not None else 1
            self.update(step)
            x = np.array(list(self.link_flow.values()))
            err = np.linalg.norm(x - x0, ord=norm) / np.sum(x0)
            x0 = x
            if err <= eps:
                break
        return self.search_step_length(x0)

    def __INT_SO(self, FFT, C, flow):
        return flow * self.LPF(FFT, C, flow)

    def __SO(self, FFT, C, flow):
        return self.LPF(FFT, C, flow) + flow * self.DER_LPF(FFT, C, flow)

    def so_opt(self, eps=5e-5, max_iter=3000, norm=2):
        self.__init_link_flow()
        x0 = np.array(list(self.link_flow.values()))
        for i in range(max_iter):
            self.__find_shortest_path_by_flow()
            direction = np.array(list(self.iter_flow.values())) - x0
            step_res = line_search(self.search_step_length, self.search_step_gradient, x0, direction)
            step = step_res[0] if step_res[0] is not None else 1
            self.update(step)
            x = np.array(list(self.link_flow.values()))
            err = np.linalg.norm(x - x0, ord=norm) / np.sum(x0)
            x0 = x
            if err <= eps:
                break
        return self.search_step_length(x0)

    def opt(self):
        if self.type == 'UE':
            return self.ue_opt()
        elif self.type == 'SO':
            return self.so_opt()

    def get_system_time_cost(self):
        if self.type == 'UE':
            time = 0
            for n, v, data in self.G.edges(data=True):
                time += (self.link_flow[(n, v)] * self.d_func(data['FFT'], data['C'], self.link_flow[(n, v)]))
            return time
        elif self.type == 'SO':
            x0 = np.array(list(self.link_flow.values()))
            return self.search_step_length(x0)
            