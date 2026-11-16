import networkx as nx
import pandas as pd
from gurobipy import GRB, Model, quicksum
import math



""" 分支流模型(Branch Flow Model, BFM)
1. 节点: [type: 'PV', 'PQ', 'slack'], [cost: float], 可选项: [g: float]
"""
class BranchFlow(object):
    def __init__(self, net: nx.DiGraph, slack_node: int):
        """
        :param net: the power grid represented by nx.DiGraph
        :type net: nx.DiGraph

        :param slack_node: the index of slack bus
        :type slack_node: int

        :note: net.node[Vmin: float, Vmax: float, g(optional): float, b(optional): float], net.edges[R: float, X:float]
        """
        self.__net = net
        self.__slack = slack_node
        self.__check_parameters()
        self.__set_default_value()

    def __check_parameters(self):
        # 必须是连通图
        if not nx.is_weakly_connected(self.__net):
            raise ValueError("Not surpport circle!")
        # 点属性检测
        for attr in ['Vmin', 'Vmax']:
            if any(attr not in attrs for _, attrs in self.__net.nodes(data=True)):
                raise ValueError(f"some nodes lack {attr} attribute!")
        # 边属性检测
        for attr in ['R', 'X']:
            if any(attr not in attrs for _, _, attrs in self.__net.edges(data=True)):
                raise ValueError(f"some edges lack {attr} attribute!")

    # 设置默认值
    def __set_default_value(self):
        for _, attrs in self.__net.nodes(data=True):
            attrs.setdefault('g', 0.0)
            attrs.setdefault('b', 0.0)

    # 设置发电节点信息
    def set_gen_and_cost(self, cost: dict[int, float]):
        self.__gen = list(cost.keys())
        self.__cost = cost
        if self.__slack not in self.__gen:
            self.__gen.append(self.__slack)
            self.__cost[self.__slack] = 0.0

    # 标准非线性模型
    def opt(self, load: dict[int, tuple[float, float]]):
        if self.__cost is None or self.__gen is None:
            raise ValueError('Please set cost and generator nodes first!')
        m = Model('nonlinear')
        m.setParam('DualReductions', 0)
        # variables
        P = m.addVars(self.__net.edges(), lb=-GRB.INFINITY, name='Send_P')
        Q = m.addVars(self.__net.edges(), lb=-GRB.INFINITY, name='Send_Q')
        v = m.addVars(
            self.__net.nodes(),
            lb={n: self.__net.nodes[n].get('Vmin') ** 2 for n in self.__net.nodes()},
            ub={n: self.__net.nodes[n].get('Vmax') ** 2 for n in self.__net.nodes()},
            name='voltage^2'
        )
        I = m.addVars(self.__net.edges(), name='current^2')
        p = m.addVars(self.__gen, lb=-GRB.INFINITY, name='gen_p')
        q = m.addVars(self.__gen, lb=-GRB.INFINITY, name='gen_q')
        # objective function (发电成本最低)
        m.setObjective(quicksum(-self.__cost[n] * p[n] for n in self.__gen), GRB.MINIMIZE)
        # constraints
        # (1). slack 节点自身约束 (p.u.)
        m.addConstr(v[self.__slack] == 1.0, 'slack')
        # (2). 功率平衡约束
        for i in self.__net.nodes():
            inflow_p = quicksum(P[j, i] - self.__net.edges[(j, i)]['R'] * I[j, i] for j in self.__net.predecessors(i))
            outflow_p = quicksum(P[i, j] for j in self.__net.successors(i))
            inflow_q = quicksum(Q[j, i] - self.__net.edges[(j, i)]['X'] * I[j, i] for j in self.__net.predecessors(i))
            outflow_q = quicksum(Q[i, j] for j in self.__net.successors(i))
            if i in self.__gen:
                m.addConstr(inflow_p - outflow_p == p[i] - self.__net.nodes[i]['g'] * v[i], name=f"balance-gen_p-{i}")
                m.addConstr(inflow_q - outflow_q == q[i] - self.__net.nodes[i]['b'] * v[i], name=f"balance-gen_q-{i}")
            elif i in load.keys():
                m.addConstr(inflow_p - outflow_p == load[i][0] - self.__net.nodes[i]['g'] * v[i], name=f'balance-load_p-{i}')
                m.addConstr(inflow_q - outflow_q == load[i][1] - self.__net.nodes[i]['b'] * v[i], name=f'balance-load_q-{i}')
            else:
                m.addConstr(inflow_p == outflow_p - self.__net.nodes[i]['g'] * v[i], name=f'balance-node_p-{i}')
                m.addConstr(inflow_q == outflow_q - self.__net.nodes[i]['b'] * v[i], name=f'balance-node_q-{i}')
        # (3). 欧姆定律
        for i, j in self.__net.edges():
            r = self.__net.edges[(i, j)]['R']
            x = self.__net.edges[(i, j)]['X']
            z = r ** 2 + x ** 2
            m.addConstr(v[i] - v[j] == 2 * (r * P[i, j] + x * Q[i, j]) - z * I[i, j], name=f'Ome-({i}, {j})')
        # (4). 支路功率约束
        m.addConstrs((I[i, j] * v[i] == P[i, j] ** 2 + Q[i, j] ** 2 for i, j in self.__net.edges()), 'branch-power')
        # 优化求解
        m.optimize()
        if m.Status == GRB.Status.INFEASIBLE:
            print("No Solution: INFISIBLE!")
        elif m.Status == GRB.Status.UNBOUNDED:
            print("No Solution: UNBOUNDED!")
        elif m.Status == GRB.Status.OPTIMAL:
            print(f"obj = {m.ObjVal}")
            for i in self.__gen:
                print(f"gen_node {i}: p = {p[i].X}, q = {q[i].X}")
            # 边的数据
            edge_data = pd.DataFrame()
            edge_data['edge'] = [f'{i}->{j}' for i, j in self.__net.edges()]
            edge_data['P'] = [P[i, j].X for i, j in self.__net.edges()]
            edge_data['Q'] = [Q[i, j].X for i, j in self.__net.edges()]
            edge_data['I'] = [I[i, j].X for i, j in self.__net.edges()]
            edge_data.set_index('edge', inplace=True)
            edge_data.to_csv('res_branch.csv')
            node_data = pd.DataFrame()
            node_data['node'] = list(self.__net.nodes())
            node_data['v'] = [math.sqrt(v[i].X) for i in self.__net.nodes()]
            node_data.set_index('node', inplace=True)
            node_data.to_csv('res_bus.csv')
    
    # 二阶锥松弛模型 (对于辐射状网络是精确的松弛)
    def opt_conic_relax(self, load: dict[int, tuple[float, float]]):
        if self.__cost is None or self.__gen is None:
            raise ValueError('Please set cost and generator nodes first!')
        m = Model('conic_relax')
        m.setParam('DualReductions', 0)
        # variables
        P = m.addVars(self.__net.edges(), lb=-GRB.INFINITY, name='Send_P')
        Q = m.addVars(self.__net.edges(), lb=-GRB.INFINITY, name='Send_Q')
        v = m.addVars(
            self.__net.nodes(),
            lb={n: self.__net.nodes[n].get('Vmin') ** 2 for n in self.__net.nodes()},
            ub={n: self.__net.nodes[n].get('Vmax') ** 2 for n in self.__net.nodes()},
            name='voltage^2'
        )
        I = m.addVars(self.__net.edges(), name='current^2')
        p = m.addVars(self.__gen, lb=-GRB.INFINITY, name='gen_p')
        q = m.addVars(self.__gen, lb=-GRB.INFINITY, name='gen_q')
        # objective function (发电成本最低)
        m.setObjective(quicksum(-self.__cost[n] * p[n] for n in self.__gen), GRB.MINIMIZE)
        # constraints
        # (1). slack 节点自身约束
        m.addConstr(v[self.__slack] == 1.0, 'slack')
        # (2). 功率平衡约束
        for i in self.__net.nodes():
            inflow_p = quicksum(P[j, i] - self.__net.edges[(j, i)]['R'] * I[j, i] for j in self.__net.predecessors(i))
            outflow_p = quicksum(P[i, j] for j in self.__net.successors(i))
            inflow_q = quicksum(Q[j, i] - self.__net.edges[(j, i)]['X'] * I[j, i] for j in self.__net.predecessors(i))
            outflow_q = quicksum(Q[i, j] for j in self.__net.successors(i))
            if i in self.__gen:
                m.addConstr(inflow_p - outflow_p == p[i] - self.__net.nodes[i]['g'] * v[i], name=f"balance-gen_p-{i}")
                m.addConstr(inflow_q - outflow_q == q[i] - self.__net.nodes[i]['b'] * v[i], name=f"balance-gen_q-{i}")
            elif i in load.keys():
                m.addConstr(inflow_p - outflow_p == load[i][0] - self.__net.nodes[i]['g'] * v[i], name=f'balance-load_p-{i}')
                m.addConstr(inflow_q - outflow_q == load[i][1] - self.__net.nodes[i]['b'] * v[i], name=f'balance-load_q-{i}')
            else:
                m.addConstr(inflow_p == outflow_p - self.__net.nodes[i]['g'] * v[i], name=f'balance-node_p-{i}')
                m.addConstr(inflow_q == outflow_q - self.__net.nodes[i]['b'] * v[i], name=f'balance-node_q-{i}')
        # (3). 欧姆定律
        for i, j in self.__net.edges():
            r = self.__net.edges[(i, j)]['R']
            x = self.__net.edges[(i, j)]['X']
            z = r ** 2 + x ** 2
            m.addConstr(v[i] - v[j] == 2 * (r * P[i, j] + x * Q[i, j]) - z * I[i, j], name=f'Ome-({i}, {j})')
        # (4). 支路功率约束 (松弛版)
        m.addConstrs((I[i, j] * v[i] >= P[i, j] ** 2 + Q[i, j] ** 2 for i, j in self.__net.edges()), 'branch-power-relax')
        # 优化求解
        m.optimize()
        if m.Status == GRB.Status.INFEASIBLE:
            print("No Solution: INFISIBLE!")
        elif m.Status == GRB.Status.UNBOUNDED:
            print("No Solution: UNBOUNDED!")
        elif m.Status == GRB.Status.OPTIMAL:
            print(f"obj = {m.ObjVal}")
            for i in self.__gen:
                print(f"gen_node {i}: p = {p[i].X}, q = {q[i].X}")
            # 边的数据
            edge_data = pd.DataFrame()
            edge_data['edge'] = [f'{i}->{j}' for i, j in self.__net.edges()]
            edge_data['P'] = [P[i, j].X for i, j in self.__net.edges()]
            edge_data['Q'] = [Q[i, j].X for i, j in self.__net.edges()]
            edge_data['I'] = [I[i, j].X for i, j in self.__net.edges()]
            edge_data.set_index('edge', inplace=True)
            edge_data.to_csv('res_conic_branch.csv')
            node_data = pd.DataFrame()
            node_data['node'] = list(self.__net.nodes())
            node_data['v'] = [math.sqrt(v[i].X) for i in self.__net.nodes()]
            node_data.set_index('node', inplace=True)
            node_data.to_csv('res_conic_bus.csv')

    # 线性近似模型
    def opt_linear_approx(self, load: dict[int, tuple[float, float]]):
        if self.__cost is None or self.__gen is None:
            raise ValueError('Please set cost and generator nodes first!')
        m = Model('linear_approx')
        m.setParam('DualReductions', 0)
        # variables
        P = m.addVars(self.__net.edges(), lb=-GRB.INFINITY, name='Send_P')
        Q = m.addVars(self.__net.edges(), lb=-GRB.INFINITY, name='Send_Q')
        v = m.addVars(
            self.__net.nodes(),
            lb={n: self.__net.nodes[n].get('Vmin') ** 2 for n in self.__net.nodes()},
            ub={n: self.__net.nodes[n].get('Vmax') ** 2 for n in self.__net.nodes()},
            name='voltage^2'
        )
        p = m.addVars(self.__gen, lb=-GRB.INFINITY, ub=0.0, name='gen_p')
        q = m.addVars(self.__gen, lb=-GRB.INFINITY, ub=0.0, name='gen_q')
        # objective function (发电成本最低)
        m.setObjective(quicksum(-self.__cost[n] * p[n] for n in self.__gen), GRB.MINIMIZE)
        # constraints
        # (1). slack 节点自身约束
        m.addConstr(v[self.__slack] == 1.0, 'slack')
        # (2). 功率平衡约束
        for i in self.__net.nodes():
            inflow_p = quicksum(P[j, i] for j in self.__net.predecessors(i))
            outflow_p = quicksum(P[i, j] for j in self.__net.successors(i))
            inflow_q = quicksum(Q[j, i] for j in self.__net.predecessors(i))
            outflow_q = quicksum(Q[i, j] for j in self.__net.successors(i))
            if i in self.__gen:
                m.addConstr(inflow_p - outflow_p == p[i], name=f"balance-gen_p-{i}")
                m.addConstr(inflow_q - outflow_q == q[i], name=f"balance-gen_q-{i}")
            elif i in load.keys():
                m.addConstr(inflow_p - outflow_p == load[i][0], name=f'balance-load_p-{i}')
                m.addConstr(inflow_q - outflow_q == load[i][1], name=f'balance-load_q-{i}')
            else:
                m.addConstr(inflow_p == outflow_p, name=f'balance-node_p-{i}')
                m.addConstr(inflow_q == outflow_q, name=f'balance-node_q-{i}')
        # (3). 欧姆定律
        for i, j in self.__net.edges():
            r = self.__net.edges[(i, j)]['R']
            x = self.__net.edges[(i, j)]['X']
            m.addConstr(v[i] - v[j] == 2 * (r * P[i, j] + x * Q[i, j]), name=f'Ome-({i}, {j})')
        # 优化求解
        m.optimize()
        if m.Status == GRB.Status.INFEASIBLE:
            print("No Solution: INFISIBLE!")
        elif m.Status == GRB.Status.UNBOUNDED:
            print("No Solution: UNBOUNDED!")
        elif m.Status == GRB.Status.OPTIMAL:
            print(f"obj = {m.ObjVal}")
            for i in self.__gen:
                print(f"gen_node {i}: p = {p[i].X}, q = {q[i].X}")
            # 边的数据
            edge_data = pd.DataFrame()
            edge_data['edge'] = [f'{i}->{j}' for i, j in self.__net.edges()]
            edge_data['P(p.u.)'] = [P[i, j].X for i, j in self.__net.edges()]
            edge_data['Q(p.u.)'] = [Q[i, j].X for i, j in self.__net.edges()]
            edge_data.set_index('edge', inplace=True)
            edge_data.to_csv('res_approx_branch.csv')
            node_data = pd.DataFrame()
            node_data['node'] = list(self.__net.nodes())
            node_data['v(p.u.)'] = [math.sqrt(v[i].X) for i in self.__net.nodes()]
            node_data.set_index('node', inplace=True)
            node_data.to_csv('res_approx_bus.csv')

            

    