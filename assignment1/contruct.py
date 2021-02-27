import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import collections
import pathpy
from pathpy.utils import Severity
import random


class FlatSet(object):
    def __init__(self, df):
        self.df = df
        self.xx = nx.from_pandas_edgelist(df, 'node1', 'node2', 'timestamp')
        self.pp = pathpy.TemporalNetwork()
        self.ff = df.values.tolist()
        for item in self.ff:
            self.pp.add_edge(item[0], item[1], item[2], timestamp_format='%S')
        self.nn = pathpy.Network.from_temporal_network(self.pp)
        print("set all package networks")

        self.N = self.nn.ncount()
        self.L = self.nn.ecount()
        self.average_degree = 2 * self.L / self.N
        self.standard_deviation_degree = self.standard_deviation_degree()

        self.average_hopcount = self.average_hopcount()
        self.largest_eigenvalue = self.largest_eigenvalue()

        self.assortativity = nx.degree_assortativity_coefficient(self.xx)

        self.average_clustering = nx.average_clustering(self.xx)
        self.diameter = nx.diameter(self.xx)

        # self.strenght = [
        #     (i + 1, list(self.nn.nodes.values())[i].get('outweight') + list(self.nn.nodes.values())[i].get('inweight'))
        #     for i in range(0, len(self.nn.nodes))]
        # self.degrees = [(i + 1, self.nn.degrees()[i]) for i in range(0, len(self.nn.nodes))]
        print("set everything")

    def infected_end(self, start):
        infected = [start]
        infected_len = []
        time_to = self.pp.observation_length()
        for current_t in self.pp.time:
            current_infected = infected.copy()
            for i in current_infected:
                if i in self.pp.sources[current_t]:
                    more = [e[1] for e in self.pp.sources[current_t][i]]
                    infected.extend(x for x in more if x not in infected)
            infected_len.append(len(infected))
            if len(infected) >= self.N * 0.8 and time_to >= current_t:
                time_to = current_t

        return infected_len, time_to

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


if __name__ == '__main__':
    print("go!")
    df = pd.read_excel('manufacturing_emails_temporal_network.xlsx',
                       dtype={'node1': int, 'node2': int, 'timestamp': str})
    G1 = FlatSet(df)
    # G2 is only timestamp shuffle
    df2 = df
    df2['timestamp'] = np.random.permutation(df['timestamp'].values)
    G2 = FlatSet(df2)

    # G3 is random link from G1 per time from G1
    G3List = []
    for t in df['timestamp'].values:
        random_link = random.sample(list(G1.nn.edges), 1)
        G3List.append([random_link[0][0], random_link[0][1], t])
    df3 = pd.DataFrame(G3List, columns=['node1', 'node2', 'timestamp'])
    G3 = FlatSet(df3)

    print(G1.pp)
    print(G2.pp)
    print(G3.pp)

    # this plot is weight distribution per edge hist
    ax1 = plt.subplot(311)
    ax1.hist([int(e['weight']) for e in G1.nn.edges.values()],
             bins=int(max([e['weight'] for e in G1.nn.edges.values()])))
    ax1.set_yscale('log')
    ax1.set_title("weight G1")
    ax2 = plt.subplot(312)
    ax2.hist([int(e['weight']) for e in G2.nn.edges.values()],
             bins=int(max([e['weight'] for e in G2.nn.edges.values()])))
    ax2.set_yscale('log')
    ax2.set_title("weight G2")
    ax3 = plt.subplot(313)
    ax3.hist([int(e['weight']) for e in G3.nn.edges.values()],
             bins=int(max([e['weight'] for e in G3.nn.edges.values()])))
    ax3.set_yscale('log')
    ax3.set_title("weight G3")

    plt.show()
