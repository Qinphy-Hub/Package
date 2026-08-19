import math

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

# global parameters
current_dir = os.path.dirname(__file__)


""" SiouxFalls 交通网络数据
@from: 网址 https://github.com/bstabler/TransportationNetworks/tree/master/SiouxFalls
@parameters: multi_dis [None or Number]
    None - 路段长度设置为两个坐标点的欧氏距离。
    Number - (默认=1) 原数据长度的倍数
@functions:
    get_network(): 获取networkx格式的图。
    get_demand(): 获取OD需求，其格式为{(O, D): size}。
    show_original_graph(file_dir): 展示此交通网络的图像, 可以选择存储图像于<file_dir>.
    show_highlight_nodes(nodes, file_name): 高亮部分节点，展示图像。
"""
class SiouxFalls:
    def __init__(self, multi_dis:int=None):
        self.__graph = self.__get_networkx(multi_dis)
        self.__demands = self.__get_demands()

    def descriptions(self):
        with open(os.path.join(current_dir, 'data/SiouxFalls/descriptions.txt'), 'r', encoding='utf-8') as f:
            content = f.read()
            print(content)
        return

    # 获取networkx格式的交通网络拓扑
    def __get_networkx(self, multi_dis=None):
        pd_link = pd.read_csv(os.path.join(current_dir, "./data/SiouxFalls/Link.csv"))
        pd_node = pd.read_csv(os.path.join(current_dir, "./data/SiouxFalls/Node.csv"))
        G = nx.DiGraph()
        for _, row in pd_node.iterrows():
            G.add_node(int(row['id']), pos=(float(row['pos_x']), float(row['pos_y'])))
        for _, row in pd_link.iterrows():
            p = G.nodes[int(row['O'])]['pos']
            q = G.nodes[int(row['D'])]['pos']
            distance = round(math.dist(p, q), 2)
            if multi_dis is not None:
                distance = float(row['FFT']) * multi_dis
            G.add_edge(int(row['O']), int(row['D']), FFT=float(row['FFT']), C=float(row['Capacity']), d=distance)
        return G

    # 获取OD需求及其大小：{(o, d): size, ...}
    def __get_demands(self):
        pd_ods = pd.read_csv(os.path.join(current_dir, "./data/SiouxFalls/OD.csv"))
        od_tupledict = {}
        for _, row in pd_ods.iterrows():
            od_tupledict[(int(row['o']), int(row['d']))] = float(row['demand'])
        return od_tupledict
    
    # 获取networkx格式的拓扑图
    def get_network(self) -> nx.DiGraph:
        return self.__graph
    
    # 获取OD需求：{(o, d): size, ...}
    def get_demands(self) -> dict[tuple, float]:
        return self.__demands
    
    """ 展示交通网络图
    @parameter: file_dir [None] - 直接展示图像；[str] - 输出图像文件到目录str
    @return: None
    """
    def show_original_graph(self, file_dir: str=None) -> None:
        pos = nx.get_node_attributes(self.__graph, 'pos')
        d = nx.get_edge_attributes(self.__graph, 'd')
        nx.draw(self.__graph, pos=pos, with_labels=True, node_size=200, font_size=10, font_weight='bold', node_color='lightblue')
        nx.draw_networkx_edge_labels(self.__graph, pos=pos, edge_labels=d)
        if file_dir == None:
            plt.show()
        else:
            plt.savefig(file_dir + "/SiouxFalls.png")
    
    """ 高亮部分节点，展示图像
    @parameter: nodes [list] - 需要高亮的节点索引列表
    @parameter: file_nam [None] - 直接展示图片；[str] - 输出图片到目录str
    @return: None
    """
    def show_highlight_nodes(self, nodes: list, file_name: str=None) -> None:
        plt.figure(figsize=(4, 5))
        for n in self.__graph.nodes():
            if n in nodes:
                self.__graph.nodes[n]['color'] = 'red'
            else:
                self.__graph.nodes[n]['color'] = 'lightblue'
        pos = nx.get_node_attributes(self.__graph, 'pos')
        d = nx.get_edge_attributes(self.__graph, 'd')
        colors = nx.get_node_attributes(self.__graph, 'color').values()
        nx.draw(self.__graph, pos=pos, with_labels=True, node_size=200, font_size=10, font_weight='bold', node_color=colors)
        nx.draw_networkx_edge_labels(self.__graph, pos=pos, edge_labels=d)
        if file_name is None:
            plt.show()
        else:
            plt.savefig(file_name)
    
