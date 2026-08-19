import copy

import networkx as nx
import numpy as np
from gurobipy import Model, GRB, LinExpr
from scipy.optimize import line_search
from collections.abc import Callable




class ClassicalFW(object):
    """ Frank-Wolfe algorithm based on the finding shortest path method
    :param G        networkx.DiGraph                        the transpoation network
    :param ods      dict[tuple, float]                      the OD demand based on G
    :param LPF      callable[[float, float, float], float]  link performance function[FFT, Capacity, flow] -> float
    :param INT_LPF  callable[[float, float, float], float]  the integral of LPF[FFT, Capacity, flow] -> float
    :param type     str['UE', 'SO', 'SUE']                  the categry of ue
    :param alpha    float                                   the parameter of default LPF(BRP)
    :param beta     float                                   the parameter of default LPF(BRP)
    :param R        float                                   
    """
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
        if self.type != 'UE' and self.type != 'SO' and self.type != 'SUE':
            raise ValueError("Only support type: UE, SO, SUE!")
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

    def get_time_cost(self):
        if self.type == 'UE':
            time = 0
            for n, v, data in self.G.edges(data=True):
                time += (self.link_flow[(n, v)] * self.d_func(data['FFT'], data['C'], self.link_flow[(n, v)]))
            return time
        elif self.type == 'SO':
            x0 = np.array(list(self.link_flow.values()))
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

    



    
    
    

    
    

    
    
        





""" 用户均衡
ClassicFW - 经典的针对UE的Frank-Wolfe算法实现
BalanceFW - 依赖流量均衡约束的Frank-Wolfe算法实现
"""
class ClassicFW:
    """
    G       networkx(DiGraph)       交通图, 需要以下参数: FFT(自由流通行时间), C(路段容量), d(路段长度)
    ods     dict({(o, d): demand})  OD需求及其大小
    alpha   float                   BPR函数的参数
    beta    float                   BPR函数的参数
    norm    int                     算法迭代误差控制
    # 拓展(非UE关注问题):
    M       float                   辅助变量: 汽车的形式里程
    """
    def __init__(self, G, ods, alpha=0.15, beta=4, norm=2, M=None):
        self.G = copy.deepcopy(G)   # 交通网络数据
        self.ods = ods              # OD需求
        self.alpha = alpha          # BPR函数的参数 (alpha)
        self.beta = beta            # BPR函数的参数 (beta)
        self.norm = norm            # 迭代误差函数的范数
        # 记录结果的容器
        self.flow = {}              # 每条路段的流量
        # 中间结果的容器
        self.iter_flow = {}         # 每次迭代的路段流量大小
        # 优化最短路算法
        self.__paths__ = True if len(set([t[0] for t in list(self.ods.keys())])) >= len(self.G.nodes()) else False
        # 拓展: 为选址模型提供所有限制范围内的子路径
        self.M = M
        self.link_set = {}
        if M is not None:
            self.__init_link_set()
    
    # 初始化小于M的子路段集合
    def __init_link_set(self):
        if self.M is None:
            return
        for n in self.G.nodes():
            for v in self.G.nodes():
                if n == v:
                    continue
                if nx.shortest_path_length(self.G, n, v, weight='d') <= self.M:
                    self.link_set[(n, v)] = set()
                    for p in nx.all_shortest_paths(self.G, n, v, weight='d'):
                        self.link_set[(n, v)].add(tuple(p))
    
    """ 找到所有OD之间的子路径
    算法优化: 依据不同的OD数量选择不同的寻找最短路算法
    1. 如果所有OD起点的节点数小于交通网络所有节点数, 那么每个OD找一次最短路最划算。
    2. 反之，找出地图中所有节点对之间的最短路最划算。
    """
    def __all_pairs_shortest_paths__(self):
        if self.__paths__:
            paths = nx.all_pairs_dijkstra_path(self.G, weight="weight")
            return dict(paths)
        else:
            paths = {}
            for r in self.ods.keys():
                if r[0] not in paths.keys():
                    paths[r[0]] = nx.single_source_dijkstra_path(self.G, r[0], weight="weight")
            return paths

    # 初始化每条路段上的流量
    def __init_flow(self):
        for e in self.G.edges():
            self.flow[e] = 0
            self.G.edges[e]['weight'] =  self.G.edges[e]['d']
        paths = self.__all_pairs_shortest_paths__()
        for r in self.ods.keys():
            path = paths[r[0]][r[1]]
            for i in range(len(path) - 1):
                self.flow[(path[i], path[i + 1])] += self.ods[r]
        for n1, n2, data in self.G.edges(data=True):
            data["weight"] = self.BPR(self.flow[(n1, n2)], data["FFT"], data["C"])

    # 初始化中间结果容器
    def __init_temp_variables(self):
        for e in self.G.edges():
            self.iter_flow[e] = 0

    # BPR 函数
    def BPR(self, flow, t0, c):
        return t0 * (1 + self.alpha * (flow / c) ** self.beta)

    # BPR 函数的积分形式
    def INT_BPR(self, flow, t0, c):
        return t0 * (flow + (self.alpha * c / (self.beta + 1)) * ((flow / c) ** (self.beta + 1)))

    # 依据当前权重寻找最优路径
    def shortest_path_by_flow(self):
        self.__init_temp_variables()
        paths = self.__all_pairs_shortest_paths__()
        # begin: 辅助变量计算
        if self.M is not None:
            for n in self.G.nodes():
                for v in self.G.nodes():
                    if n == v or (n, v) not in self.link_set.keys():
                        continue
                    # equivalent path
                    for p in nx.all_shortest_paths(self.G, n, v, weight='weight'):
                        l = 0
                        for i in range(len(p) - 1):
                            l += (self.G.edges[(p[i], p[i+1])]['d'])
                        if l <= self.M:
                            self.link_set[(n, v)].add(tuple(p))
        # end: 辅助变量计算
        for r in self.ods.keys():
            path = paths[r[0]][r[1]]
            for i in range(len(path) - 1):
                self.iter_flow[(path[i], path[i + 1])] += self.ods[r]

    # 寻找迭代步长的目标函数
    def objective_step(self, x):
        expr = 0
        i = 0
        for _, _, data in self.G.edges(data=True):
            expr += (self.INT_BPR(x[i], data["FFT"], data["C"]))
            i += 1
        return expr

    # 迭代方向
    def gradient_step(self, x):
        expr = []
        i = 0
        for _, _, data in self.G.edges(data=True):
            expr.append(self.BPR(x[i], data["FFT"], data["C"]))
            i += 1
        return expr

    # 根据步长更新权重(最短路径的权重依据)
    def update(self, step):
        for e in self.G.edges():
            self.flow[e] += (step * (self.iter_flow[e] - self.flow[e]))
            self.G.edges[e]['weight'] = self.BPR(self.flow[e], self.G.edges[e]["FFT"], self.G.edges[e]["C"])

    # Frank-Wolfe 方法
    def opt(self, eps=5e-5, max_iter=3000):
        self.__init_flow()
        x0 = np.array(list(self.flow.values()))
        for i in range(max_iter):
            self.shortest_path_by_flow()
            direction = np.array(list(self.iter_flow.values())) - x0
            step_res = line_search(self.objective_step, self.gradient_step, x0, direction)
            step = step_res[0]
            self.update(step)
            x = np.array(list(self.flow.values()))
            err = np.linalg.norm(x - x0, ord=self.norm) / np.sum(x0)
            x0 = x
            if err <= eps:
                break
        return self.objective_step(x0), x0




# 借助流量均衡约束, 不依赖最短路径方法
class BalanceFW(object):
    """
    G       networkx(DiGraph)       交通图, 需要以下参数: FFT(自由流通行时间), C(路段容量), d(路段长度)
    ods     dict({(o, d): demand})  OD需求及其大小
    alpha   float                   BPR函数的参数
    beta    float                   BPR函数的参数
    od_flag bool                    是否计算OD需求分配流量的结果
    norm    int                     算法迭代误差控制
    """
    def __init__(self, G, ods, alpha=0.15, beta=4, od_flag=True, norm=2):
        self.G = copy.deepcopy(G)
        self.ods = ods
        self.alpha = alpha
        self.beta = beta
        self.od_flag = od_flag
        self.norm = norm
        # 获取必要的结果信息
        self.od_flow = {}
        self.flow = {}
        # 初始化
        self.__init_data()

    # 初始化结果容器
    def __init_data(self):
        for e in self.G.edges():
            self.flow[e] = 0
        for r in self.ods.keys():
            self.od_flow[r] = {}
            for e in self.G.edges():
                self.od_flow[r][e] = 0

    # 找到初始迭代点, 与经典方法保持一致。(也可以任意取)
    def __init_flow_point__(self):
        paths = {}
        if len(set([t[0] for t in list(self.ods.keys())])) >= len(self.G.nodes()):
            paths = dict(nx.all_pairs_dijkstra_path(self.G, weight='d'))
        else:
            for r in self.ods.keys():
                if r[0] not in self.ods.keys():
                    paths[r[0]] = nx.single_source_dijkstra_path(self.G, source=r[0], weight='d')
        for r in self.ods.keys():
            path = paths[r[0]][r[1]]
            for i in range(len(path) - 1):
                self.flow[(path[i], path[i + 1])] += self.ods[r]
                if self.od_flag:
                    self.od_flow[r][(path[i], path[i + 1])] += self.ods[r]
        return np.array(list(self.flow.values()))

    # BPR 函数的积分
    def INT_BPR(self, f, FFT, C):
        return FFT * (f + (self.alpha * C / (self.beta + 1)) * ((f / C) ** (self.beta + 1)))

    # BPR 函数
    def BPR(self, f, FFT, C):
        return FFT * (1 + self.alpha * ((1.0 * f / C) ** self.beta))

    # 线性化之后的目标函数
    def objective_linear(self, x, x0):
        expr = LinExpr()
        i = 0
        for e in self.G.edges():
            expr += (x[e] * self.BPR(x0[i], self.G.edges[e]['FFT'], self.G.edges[e]['C']))
            i += 1
        return expr

    # 线搜索(line search)查找步长：目标函数
    def objective_step(self, x):
        expr = 0
        i = 0
        for _, _, data in self.G.edges(data=True):
            expr += (self.INT_BPR(x[i], data["FFT"], data["C"]))
            i += 1
        return expr

    # 线搜索(line search)查找步长：目标函数的方向向量
    def gradient_step(self, x):
        expr = []
        i = 0
        for _, _, data in self.G.edges(data=True):
            expr.append(self.BPR(x[i], data["FFT"], data["C"]))
            i += 1
        return expr

    # Frank-Wolfe 方法
    def opt(self, eps=5e-5, max_iter=2148):
        x0 = self.__init_flow_point__()
        R = list(self.ods.keys())
        for i in range(max_iter):
            # 线性化之后的模型求解
            m = Model('linear')
            m.setParam('OutputFlag', 0)
            x = m.addVars(self.G.edges(), R, vtype=GRB.CONTINUOUS, lb=0, name="x")
            x_a = m.addVars(self.G.edges(), vtype=GRB.CONTINUOUS, lb=0, name="x_a")
            m.setObjective(self.objective_linear(x_a, x0), GRB.MINIMIZE)
            for r1, r2 in R:
                name_title = '(' + str(r1) + ',' + str(r2) + ')'
                m.addConstr(x.sum(r1, '*', r1, r2) == self.ods[(r1, r2)], name='origin_out' + name_title)
                m.addConstr(x.sum('*', r1, r1, r2) == 0, name='origin_in' + name_title)
                m.addConstr(x.sum(r2, '*', r1, r2) == 0, name='destination_out' + name_title)
                m.addConstr(x.sum('*', r2, r1, r2) == self.ods[(r1, r2)], name='destination_in' + name_title)
                for node in self.G.nodes():
                    if node == r1 or node == r2:
                        continue
                    m.addConstr(x.sum(node, '*', r1, r2) - x.sum('*', node, r1, r2) == 0, name="middle_" + name_title + "_" + str(node))
            for e1, e2 in self.G.edges():
                m.addConstr(x.sum(e1, e2, '*') == x_a.sum(e1, e2), name=str((e1, e2)))
            m.update()
            m.optimize()
            if m.status != GRB.Status.OPTIMAL:
                print('Optimal solution not found!')
                break
            # 线性模型求解结果用于判断迭代方向
            y = []
            for e in self.G.edges():
                y.append(x_a[e].X)
            y = np.array(y)
            # 使用线搜索求解迭代步长
            step_res = line_search(f=self.objective_step, myfprime=self.gradient_step, xk=x0, pk=y-x0)
            step = step_res[0]
            # 更新参数
            x1 = x0 + step * (y - x0)
            err = np.linalg.norm(x1 - x0, ord=self.norm) / np.sum(x0)
            # 更新迭代参数
            x0 = x1
            if self.od_flag:
                for r1, r2 in R:
                    for e1, e2 in self.G.edges():
                        self.od_flow[(r1, r2)][(e1, e2)] = (self.od_flow[(r1, r2)][(e1, e2)] + step * (x[e1, e2, r1, r2].X - self.od_flow[(r1, r2)][(e1, e2)]))
            if err < eps:
                print('limited by eps. Iter =', i + 1)
                break
        return self.objective_step(x0)