import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.mlab as mlab
import scipy
import networkx as nx
import collections
import pathpy
from pathpy.utils import Severity
import random
import pickle


class FlatSet(object):
    def __init__(self, df):
        self.df = df
        self.xx = nx.from_pandas_edgelist(df, 'node1', 'node2', 'timestamp')
        self.pp = pathpy.TemporalNetwork()
        self.ff = df.values.tolist()
        for item in self.ff:
            self.pp.add_edge(item[0], item[1], item[2], directed=False, timestamp_format='%S')
        self.nn = pathpy.Network.from_temporal_network(self.pp, directed=False)
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

        self.intervals = []
        print("set everything")

    def edge_intervals(self):
        for edgeval in self.nn.edges:
            current_intervals = []
            for edgeval_current in self.pp.tedges[::2]:
                if edgeval[0] == edgeval_current[0] and edgeval[1] == edgeval_current[1]:
                    current_intervals.append(edgeval_current[2])
                    #print(edgeval_current)

            #current_intervals.pop(0)  # do this to remove the edge itself, e.g. [a, a] is no interval
            #print(current_intervals)

            current_intervals.sort()
            self.intervals.extend(list(np.diff(current_intervals)))
            #for x, y in zip(current_intervals[0::], current_intervals[1::]):
            #    self.intervals.append(y - x)
        print(self.intervals)

        #print("G len(intervals): " + str(len(self.intervals)))

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

    #compute probability density for G1
    ####################################################################
    G1.edge_intervals()

    # intervals_g1 = []
    #
    # for edgeval in G1.nn.edges:
    #     current_intervals = []
    #     for edgeval_current in G1.pp.tedges[::2]:
    #         if edgeval[0] == edgeval_current[0] and edgeval[1] == edgeval_current[1]:
    #             current_intervals.append(edgeval_current[2])
    #
    #     current_intervals.pop(0) # do this to remove the edge itself, e.g. [a, a] is no interval
    #     #print(current_intervals)
    #
    #     for x, y in zip(current_intervals[0::], current_intervals[1::]):
    #         intervals_g1.append(y-x)
    #     #print(intervals)
    #
    #     print("G1 len(intervals): " + str(len(intervals_g1)))
    ####################################################################


    #compute probability density for G2
    ####################################################################
    G2.edge_intervals()
    # intervals_g2 = []
    #
    # for edgeval in G2.nn.edges:
    #     current_intervals = []
    #     for edgeval_current in G2.pp.tedges[::2]:
    #         if edgeval[0] == edgeval_current[0] and edgeval[1] == edgeval_current[1]:
    #             current_intervals.append(edgeval_current[2])
    #
    #     current_intervals.pop(0) # do this to remove the edge itself, e.g. [a, a] is no interval
    #     #print(current_intervals)
    #
    #     for x, y in zip(current_intervals[0::], current_intervals[1::]):
    #         intervals_g2.append(y-x)
    #     #print(intervals)
    #
    #     print("G2 len(intervals): " + str(len(intervals_g2)))
    ####################################################################


    #compute probability density for G3
    ####################################################################
    G3.edge_intervals()
    # intervals_g3 = []
    #
    # for edgeval in G3.nn.edges:
    #     current_intervals = []
    #     for edgeval_current in G3.pp.tedges[::2]:
    #         if edgeval[0] == edgeval_current[0] and edgeval[1] == edgeval_current[1]:
    #             current_intervals.append(edgeval_current[2])
    #
    #     current_intervals.pop(0) # do this to remove the edge itself, e.g. [a, a] is no interval
    #     #print(current_intervals)
    #
    #     for x, y in zip(current_intervals[0::], current_intervals[1::]):
    #         intervals_g3.append(y-x)
    #     #print(intervals)
    #
    #     print("G3 len(intervals): " + str(len(intervals_g3)))
    ####################################################################

    # with open('graph_data.pkl',"wb") as graph_data:
    #     graphs = [G1.intervals, G2.intervals, G3.intervals]
    #     pickle.dump(graphs, graph_data)
    # with open('graph_data.pkl',"rb") as graph_data:
    #     graphs = pickle.load(graph_data)
    #     intervals1 = graphs[0]
    #     intervals2=graphs[1]
    #     intervals3 = graphs[2]

    number_of_bins_for_hist = 1000 #number of bins in graphs

    #plot probability density for G1
    #f1 = plt.figure(1)
    n1, x1, _ = plt.hist(G1.intervals, bins=number_of_bins_for_hist,density=False, color='r', label="G1", alpha=0.3)
    plt.xlabel("Arrival intervals")
    plt.ylabel("Frequency")
    plt.title("Arrival interval histogram")
    #f1.show()

    #plot probability density for G2
    #f2 = plt.figure(2)
    n2, x2, _ = plt.hist(G2.intervals, bins=number_of_bins_for_hist,density=False, color='g', label="G2", alpha=0.3)
    #plt.xlabel("Values")
    #plt.ylabel("Frequency")
    #plt.title("G2 - Graph 2")
    #f2.show()

    #plot probability density for G3
    #f3 = plt.figure(3)
    n3, x3, _ = plt.hist(G3.intervals, bins=number_of_bins_for_hist,density=False, color='b', label="G3", alpha=0.3)
    #plt.xlabel("Values")
    #plt.ylabel("Frequency")
    #plt.title("G3 - Graph 3")
    #f3.show()

    # bin_centers = 0.5 * (x1[1:] + x1[:-1])
    # plt.plot(bin_centers, n1, color='r', alpha = 0.5, linestyle="", marker=',')
    # bin_centers = 0.5 * (x2[1:] + x2[:-1])
    # plt.plot(bin_centers, n2, color='g', alpha = 0.5, linestyle="", marker=',')
    # bin_centers = 0.5 * (x3[1:] + x3[:-1])
    # plt.plot(bin_centers, n3, color='b', alpha = 0.5, linestyle="", marker='o', s=1 )

    #plt.ylim(0, 10000)
    plt.yscale('log')
    plt.legend()
    plt.show()