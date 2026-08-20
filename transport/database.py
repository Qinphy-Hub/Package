import math

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

# global parameters
current_dir = os.path.dirname(__file__)


""" SiouxFalls Traffic Network Data
@from: website https://github.com/bstabler/TransportationNetworks/tree/master/SiouxFalls
@parameters: multi_dis [None or Number]
    None - The line segment length is set to the Euclidean distance between the two coordinate points.
    Number - Multiple of the original data length.
@functions:
    get_network(): get the traffic network.
    get_demand(): get od demand, format: {(O, D): size}.
    show_original_graph(file_dir): show the picture of this traffic network, opt: save picture file to <file_dir>.
    show_highlight_nodes(nodes, file_name): highlight some nodes, opt: save picture file to <file_dir>.
"""
class SiouxFalls:
    def __init__(self):
        self.__graph = self.__get_networkx()
        self.__demands = self.__get_demands()

    def descriptions(self):
        with open(os.path.join(current_dir, 'data/SiouxFalls/descriptions.txt'), 'r', encoding='utf-8') as f:
            content = f.read()
            print(content)
        return

    # get the networkx format topology data
    def __get_networkx(self):
        pd_link = pd.read_csv(os.path.join(current_dir, "./data/SiouxFalls/Link.csv"))
        pd_node = pd.read_csv(os.path.join(current_dir, "./data/SiouxFalls/Node.csv"))
        G = nx.DiGraph()
        for _, row in pd_node.iterrows():
            G.add_node(int(row['id']), pos=(float(row['pos_x']), float(row['pos_y'])))
        for _, row in pd_link.iterrows():
            G.add_edge(int(row['O']), int(row['D']), FFT=float(row['FFT']), C=float(row['Capacity']), d=float(row['FFT']))
        return G

    # get OD demand and its size, {(o, d): size, ...}
    def __get_demands(self):
        pd_ods = pd.read_csv(os.path.join(current_dir, "./data/SiouxFalls/OD.csv"))
        od_tupledict = {}
        for _, row in pd_ods.iterrows():
            od_tupledict[(int(row['o']), int(row['d']))] = float(row['demand'])
        return od_tupledict
    
    # get networkx format data
    def get_network(self) -> nx.DiGraph:
        return self.__graph
    
    # get od demand: {(o, d): size, ...}
    def get_demands(self) -> dict[tuple, float]:
        return self.__demands
    
    """ show the picture of traffic network
    @parameter: file_dir [None] - show picture directly; [str] - output file to <str>.
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
    
    """ highlight some nodes
    @parameter: nodes [list] - the nodes which need to highlight.
    @parameter: file_nam [None] - show picture directly; [str] - output file to <str>.
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
    
