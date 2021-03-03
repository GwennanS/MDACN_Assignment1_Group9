import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import collections
import pathpy


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


class FlatSet(object):
    def __init__(self, df):
        self.xx = nx.from_pandas_edgelist(df, 'node1', 'node2', 'timestamp')
        self.pp = pathpy.TemporalNetwork()
        self.ff = df.values.tolist()
        for item in self.ff:
            self.pp.add_edge(item[0], item[1], item[2], timestamp_format='%S')
        self.nn = pathpy.Network.from_temporal_network(self.pp, directed=False)
        print("set all package networks")

        self.N = self.nn.ncount()
        self.L = self.nn.ecount()
        self.average_degree = np.mean(self.nn.degrees())
        self.standard_deviation_degree = np.std(self.nn.degrees())
        print("set N, L, average degree and standard deviation degree")

        self.average_hopcount = self.average_hopcount()
        self.largest_eigenvalue = self.largest_eigenvalue()
        print("set average_hopcount and largest_eigenvalue")

        self.assortativity = nx.degree_assortativity_coefficient(self.xx)
        print("set assortativity")

        self.average_clustering = nx.average_clustering(self.xx)
        self.diameter = nx.diameter(self.xx)
        print("set average_clustering and diameter")

        # self.weights = zip(self.nn.edges, [e['weight'] for e in self.nn.edges.values()])
        # self.degrees = zip(self.nn.nodes, self.nn.degrees())

    def average_hopcount(self):
        pathlengths = []
        for v in self.xx.nodes():
            spl = dict(nx.single_source_shortest_path_length(self.xx, v))
            for p in spl:
                pathlengths.append(spl[p])
        return sum(pathlengths) / len(pathlengths)

    def standard_deviation_degree(self):
        sum = 0
        for degree in self.nn.degrees():
            sum += (degree - self.average_degree) ** 2
        return (sum / self.N - 1) ** 0.5

    def largest_eigenvalue(self):
        L = nx.normalized_laplacian_matrix(self.xx)
        e = np.linalg.eigvals(L.A)
        return max(e)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('Complex Networks')
    df = pd.read_excel('manufacturing_emails_temporal_network.xlsx',
                       dtype={'node1': int, 'node2': int, 'timestamp': str})
    flat_set = FlatSet(df)

    print("N: ", flat_set.N)
    print("L: ", flat_set.L)
    print("average degree: ", flat_set.average_degree)
    print("standard deviation of the degree: ", flat_set.standard_deviation_degree)
    print("degree correlation (assortativity): ", flat_set.assortativity)
    print("clustering coefficient: ", flat_set.average_clustering)
    print("average hopcount: ", flat_set.average_hopcount)
    print("diameter: ", flat_set.diameter)
    print("largest eigenvalue:", flat_set.largest_eigenvalue)

    # plot degree hist and weight hist
    ax2 = plt.subplot(211)
    ax2.hist(flat_set.nn.degrees(), 150)
    ax2.set_title("Degree distribution")
    ax2.set_xlabel('degree')
    ax2.set_ylabel('num of nodes')

    ax1 = plt.subplot(212)
    ax1.set_xlim([-1, 420])
    ax1.hist([int(e['weight']) for e in flat_set.nn.edges.values()],
             bins=int(max([e['weight'] for e in flat_set.nn.edges.values()])))
    ax1.set_yscale('log')
    ax1.set_xlabel('weight')
    ax1.set_ylabel('num of links')
    ax1.set_title("Link weights distribution")

    plt.show()

# degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
# degreeCount = collections.Counter(degree_sequence)
# deg, cnt = zip(*degreeCount.items())
#
# fig, ax = plt.subplots()
# plt.bar(deg, cnt, width=0.80, color="b")
#
# plt.title("Degree Histogram")
# plt.ylabel("Count")
# plt.xlabel("Degree")
# ax.set_xticks([d + 0.4 for d in deg])
# ax.set_xticklabels(deg)
#
# # draw graph in inset
# plt.axes([0.4, 0.4, 0.5, 0.5])
# Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
# pos = nx.spring_layout(G)
# plt.axis("off")
# nx.draw_networkx_nodes(G, pos, node_size=20)
# nx.draw_networkx_edges(G, pos, alpha=0.4)
# plt.show()
#
# print("N: ", len(G.nodes))
# print("L: ", len(G.edges))
# print("average degree: ", get_average_degree(G))
# print("standard deviation of the degree: ", standard_deviation_degree(G))
# print("degree correlation (assortativity): ", nx.degree_assortativity_coefficient(G))
# print("clustering coefficient: ", nx.average_clustering(G))
# print("average hopcount: ", average_hopcount(G))
# print("diameter: ", nx.diameter(G))
# print("largest eigenvalue:", largest_eignevalue(G))

