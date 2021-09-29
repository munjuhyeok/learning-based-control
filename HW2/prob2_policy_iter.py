# -*- coding: utf-8 -*-
from argparse import ArgumentParser, RawTextHelpFormatter
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import textwrap


def plot_graph(adjacency_matrix, path=None):
    adjacency_matrix = np.array(adjacency_matrix)
    rows, cols = np.where(adjacency_matrix > 0)
    edges = list(zip(rows.tolist(), cols.tolist()))
    values = [adjacency_matrix[i][j] for i, j in edges]
    weighted_edges = [(e[0], e[1], values[idx]) for idx, e in enumerate(edges)]
    plt.cla()
    fig = plt.figure(1)

    plt.title("Korea highway map")
    G = nx.Graph()
    G.add_weighted_edges_from(weighted_edges)
    # plot
    labels = nx.get_edge_attributes(G, 'weight')
    pos_map = [
        [223, -85],
        [137, -104],
        [226, -213],
        [262, -269],
        [445, -200],
        [538, -26],
        [155, -245],
        [310, -377],
        [490, -420],
        [180, -480],
        [200, -688],
        [385, -578],
        [550, -500],
        [700, -434],
        [113, -756],
        [357, -752],
        [450, -672],
        [630, -675],
        [684, -572],
    ]
    # city_list = ['서울','인천','평택','천안','제천','강릉','당진','대전','김천','군산','광주','함양','대구','포항','목포','여수','진주','부산','울산']
    city_list = ['Seoul', 'Incheon', 'Pyeongtaek', 'Cheonan', 'Jecheon', 'Gangneung', 'Dangjin', 'Daejeon', 'Gimcheon',
                 'Gunsan',
                 'Gwangju', 'Hamyang', ' Daegu', 'Pohang', 'Mokpo', 'Yeosu', 'Jinju', 'Busan', 'Ulsan']
    city_list = {i: city_list[i] for i in range(len(pos_map))}
    pos = {i: pos_map[i] for i in range(len(pos_map))}
    nx.draw(G, labels=city_list, pos=pos, with_labels=True, font_size=10)
    nodes = nx.draw_networkx_nodes(G, pos, node_color='#8FC2FF')
    nodes.set_edgecolor('white')
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=labels, font_size=8)

    if path is not None:
        policy_edge = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        nx.draw_networkx_edges(G, pos, edgelist=policy_edge, edge_color='r', width=2)

    fig.set_size_inches(10, 10)
    plt.gca().set_aspect('equal')
    plt.show()


def print_optim_path(optim_path):
    optim_path_info = ["{}->{}".format(optim_path[i], optim_path[i + 1]) for i in range(len(optim_path) - 1)]
    return ", ".join(optim_path_info)


def get_optim_policy(D=None, optim_value=None, depart_pos=None, terminal_pos=None, gamma=None):
    get_optim_policy = []
    # Todo
    return get_optim_policy


def get_optim_path(D=None, optim_value=None, depart_pos=None, terminal_pos=None, gamma=None):
    optim_path = []
    # Todo
    return optim_path


def get_optim_value(D=None, threshold=0.001, gamma=0.9, depart_pos=7, terminal_pos=0):
    optim_value = []
    # Todo
    return optim_value


if __name__ == '__main__':
    parser = ArgumentParser(
        prog='prob2_policy_iter.py',
        formatter_class=RawTextHelpFormatter,
        epilog=textwrap.dedent('''\
        City List : 
         'Seoul',         0
         'Incheon',       1
         'Pyeongtaek',    2
         'Cheonan',       3
         'Jecheon',       4
         'Gangneung',     5
         'Dangjin',       6
         'Daejeon',       7
         'Gimcheon',      8
         'Gunsan',        9
         'Gwangju',       10
         'Hamyang',       11
         'Daegu',         12
         'Pohang',        13
         'Mokpo',         14
         'Yeosu',         15
         'Jinju',         16
         'Busan',         17
         'Ulsan',         18
         ''')
    )

    parser.add_argument("-d", "--depart", help="departing city(default Daejeon)", type=str, default="7")
    parser.add_argument("-t", "--terminal", help="terminalal city(default Seoul)", type=str, default="0")

    args = parser.parse_args()

    D = np.genfromtxt('HW2_adjacency_matrix.csv', delimiter=',')
    D = D.astype(int)

    num_nodes = len(D)
    depart_pos = int(args.depart)
    terminal_pos = int(args.terminal)
    gamma = 0.9

    optim_value = get_optim_value(D, threshold=0.001, gamma=gamma, depart_pos=depart_pos, terminal_pos=terminal_pos)
    optim_policy = get_optim_policy(D, optim_value, depart_pos, terminal_pos, gamma)
    optim_path = get_optim_path(D, optim_value, depart_pos, terminal_pos, gamma)

    print("-" * 20)
    print("The value of states using policy_iteration")
    print("{}".format(list(np.around(optim_value, decimals=2))))
    print("-" * 20)
    print("The best action for every node")
    print(optim_policy)
    print("-" * 20)
    print("The best action from departure to the terminal")
    print(optim_path)
    print(print_optim_path(optim_path))

    plot_graph(D, path=optim_path)
