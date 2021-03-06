import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
import collections
import pathpy
from pathpy.utils import Severity


class FlatSet(object):
    def __init__(self, df):
        self.xx = nx.from_pandas_edgelist(df, 'node1', 'node2', 'timestamp')
        self.pp = pathpy.TemporalNetwork()
        self.ff = df.values.tolist()
        for item in self.ff:
            self.pp.add_edge(item[0], item[1], item[2], timestamp_format='%S')
        self.nn = pathpy.Network.from_temporal_network(self.pp)
        print("set all package networks")

        self.betweenness = self.betweenness(self.nn)
        self.closeness = self.closeness(self.nn)

        self.N = self.nn.ncount()
        self.L = self.nn.ecount()
        self.average_degree = 2 * self.L / self.N
        self.standard_deviation_degree = self.standard_deviation_degree()

        self.average_hopcount = self.average_hopcount()
        self.largest_eigenvalue = self.largest_eigenvalue()

        self.assortativity = nx.degree_assortativity_coefficient(self.xx)

        self.average_clustering = nx.average_clustering(self.xx)
        self.diameter = nx.diameter(self.xx)

        self.strenght = [
            (i + 1, list(self.nn.nodes.values())[i].get('outweight') + list(self.nn.nodes.values())[i].get('inweight'))
            for i in range(0, len(self.nn.nodes))]
        self.degrees = [(i + 1, self.nn.degrees()[i]) for i in range(0, len(self.nn.nodes))]
        print("set everything")

    def infected_end(self, start):
        infected = [start]
        infected_len = []
        time_to = self.pp.observation_length()
        time_sum = 0
        for current_t in self.pp.time:
            current_infected = infected.copy()
            for i in current_infected:
                if i in self.pp.sources[current_t]:
                    more = [e[1] for e in self.pp.sources[current_t][i]]
                    infected.extend(x for x in more if x not in infected)
                if i in self.pp.targets[current_t]:
                    more = [e[0] for e in self.pp.targets[current_t][i]]
                    infected.extend(x for x in more if x not in infected)
            infected_len.append(len(infected))
            if len(infected) >= self.N * 0.8 and time_to >= current_t:
                time_to = current_t
            if len(infected) < self.N * 0.8:
                time_sum = time_sum + current_t * (len(infected) - len(current_infected))

        if time_to == self.pp.observation_length():
            time_sum = self.pp.observation_length()
        else:
            time_sum = time_sum / self.N * 0.8

        return infected_len, time_to, time_sum

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

    def betweenness(self, nw):
        return list(pathpy.algorithms.centralities.betweenness(nw).items())

    def closeness(self, nw):
        return list(pathpy.algorithms.centralities.closeness(nw).items())



# def sample_paths_from_temporal_network_dag(tempnet, delta=1, max_subpath_length=None):
#     """
#     Estimates the frequency of causal paths in a temporal network assuming a
#     maximum temporal distance of delta between consecutive
#     time-stamped links on a path. This method first creates a directed and acyclic
#     time-unfolded graph based on the given parameter delta. This directed acyclic
#     graph is used to calculate causal paths for a given delta, randomly sampling num_roots
#     roots in the time-unfolded DAG.
#
#     Parameters
#     ----------
#     tempnet : pathpy.TemporalNetwork
#         TemporalNetwork to extract the time-respecting paths from
#     delta : int
#         Indicates the maximum temporal distance up to which time-stamped
#         links will be considered to contribute to a causal path.
#         For (u,v;3) and (v,w;7) a causal path (u,v,w) is generated
#         for 0 < delta <= 4, while no causal path is generated for
#         delta > 4. Every time-stamped edge is a causal path of
#         length one. Default value is 1.
#     num_roots : int
#         The number of randomly chosen roots that will be used to estimate path statistics.
#
#     Returns
#     -------
#     Paths
#         An instance of the class Paths, which can be used to generate higher- and multi-order
#         models of causal paths in temporal networks.
#     """
#     # generate a single time-unfolded DAG
#     pathpy.utils.Log.set_min_severity(Severity.WARNING)
#     dag, node_map = pathpy.DAG.from_temporal_network(tempnet, delta)
#     # dag.topsort()
#     # assert dag.is_acyclic
#     # print(dag)
#     infect_num = []
#     causal_paths = pathpy.Paths()
#
#     # For each root in the time-unfolded DAG, we generate a
#     # causal tree and use it to count all causal paths
#     # that originate at this root
#     print(dag)
#     current_root = 1
#
#     for root in dag.roots:
#         causal_tree, causal_mapping = pathpy.path_extraction.generate_causal_tree(dag, root, node_map)
#         #    if num_roots > 10:
#         #        step = num_roots / 10
#         #        if current_root % step == 0:
#         #            print('Analyzing tree {0}/{1} ...'.format(current_root, num_roots))
#
#         # calculate all unique longest path in causal tree
#         causal_paths = pathpy.path_extraction.paths_from_dag(causal_tree, causal_mapping, repetitions=False,
#                                                              max_subpath_length=max_subpath_length)
#         infect_num.append(len(causal_paths.nodes))
#         print(causal_paths.nodes)
#         current_root += 1
#     #    current_root += 1
#
#     return infect_num


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('Complex Networks')
    df = pd.read_excel('manufacturing_emails_temporal_network.xlsx',
                       dtype={'node1': int, 'node2': int, 'timestamp': str})
    flat_set = FlatSet(df)

    infect_num = []  # This contains I(t) for all nodes
    infect_num_rate = []  # this contains list of time to reach 0.8N for all nodes
    infect_num_avg = []  # this contains the avg time to reach 0.8N for all nodes
    # for n in range(1, 3):
    for n in flat_set.pp.nodes:
        list_n, time_n, sum_n = flat_set.infected_end(n)
        infect_num.append(list_n)
        infect_num_rate.append((n, time_n))
        infect_num_avg.append((n, sum_n))
        print(n)

    min_time = min([len(item) for item in infect_num])  # this is so index always works
    infect_num_expect = []  # this is E[I(t)]
    infect_num_expect_min = []  # this is E[I(t)] -  sqrt(Var[I(t)]))
    infect_num_expect_plus = []  # this is E[I(t)] +  sqrt(Var[I(t)]))
    for t in range(0, min_time):
        mean = np.mean([item[int(t)] for item in infect_num])
        var = np.std([item[int(t)] for item in infect_num])
        # infect_num_expectN = sum([item[t] for item in infect_num])/len(infect_num)
        infect_num_expect.append(mean)
        infect_num_expect_min.append(mean - var)
        infect_num_expect_plus.append(mean + var)

    # this is R' rank
    infect_num_avg = sorted(infect_num_avg, key=lambda x: x[1])
    ranking_avg = [None] * flat_set.N
    rank = 0
    for tuple_avg in infect_num_avg:
        ranking_avg[rank] = tuple_avg[0]
        rank = rank + 1
    print(infect_num_avg)
    print("ranking_avg: ", ranking_avg)

    # this is R rank
    infect_num_rate = sorted(infect_num_rate, key=lambda x: x[1])
    ranking_infect = [None] * flat_set.N
    rank = 0
    for tuple_rate in infect_num_rate:
        ranking_infect[rank] = tuple_rate[0]
        rank = rank + 1
    print(infect_num_rate)
    print("ranking_infect: ", ranking_infect)

    # this is D rank
    ranking_degree = [None] * flat_set.N
    rank = 0
    degree_sort = sorted(flat_set.degrees, key=lambda x: -x[1])
    for tuple_degree in degree_sort:
        ranking_degree[rank] = tuple_degree[0]
        rank = rank + 1
    print(degree_sort)
    print("ranking_degree: ", ranking_degree)

    # this is S rank
    ranking_strenght = [None] * flat_set.N
    rank = 0
    strenght = sorted(flat_set.strenght, key=lambda x: -x[1])
    for tuple_rate in strenght:
        ranking_strenght[rank] = tuple_rate[0]
        rank = rank + 1
    print(strenght)
    print("ranking_strenght: ", ranking_strenght)


    # this is closeness
    ranking_closeness = [None] * flat_set.N
    rank = 0
    closeness = sorted(flat_set.closeness, key=lambda x: -x[1])
    print(closeness)
    for tuple_cls in closeness:
        ranking_closeness[rank] = tuple_cls[0]
        rank = rank + 1
    print("ranking closeness", ranking_closeness)

    ranking_betweenness = [None] * flat_set.N
    rank = 0
    betweenness = sorted(flat_set.betweenness, key=lambda x: -x[1])
    print(betweenness)
    for tuple_btw in betweenness:
        ranking_betweenness[rank] = tuple_btw[0]
        rank = rank + 1
    print("ranking betweenness", ranking_betweenness)

    # uncomment this to plot centrality metric
    centrality_metric = []
    for f in np.linspace(0.05, 0.5, 10):
        size_f = int(f * flat_set.N)
        degF = len(set(ranking_infect[0:size_f]).intersection(set(ranking_degree[0:size_f]))) / size_f
        strF = len(set(ranking_infect[0:size_f]).intersection(set(ranking_strenght[0:size_f]))) / size_f
        centrality_metric.append((degF, strF))

    p1, p2 = plt.plot(centrality_metric)
    plt.legend([p1, p2], ["Degree", "Strength"])
    plt.title("Centrality metrics")

    # uncomment this for q12 - degree, strength, betweenness and closeness
    #     centrality_metric = []
    #     for f in np.linspace(0.05, 0.5, 10):
    #         size_f = int(f * flat_set.N)
    #         degF = len(set(ranking_infect[0:size_f]).intersection(set(ranking_degree[0:size_f]))) / size_f
    #         strF = len(set(ranking_infect[0:size_f]).intersection(set(ranking_strenght[0:size_f]))) / size_f
    #         clsF = len(set(ranking_infect[0:size_f]).intersection(set(ranking_closeness[0:size_f]))) / size_f
    #         btwF = len(set(ranking_infect[0:size_f]).intersection(set(ranking_betweenness[0:size_f]))) / size_f
    #         centrality_metric.append((degF, strF, clsF, btwF))
    #
    #     p1, p2, p3, p4 = plt.plot(centrality_metric)
    #     plt.legend([p1, p2, p3, p4], ["Degree", "Strength", "Closeness", "Betweenness"])
    #     y = np.array(centrality_metric)
    #     plt.xticks(np.arange(10), [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    #     plt.yticks(np.arange(y.min(), y.max(), 0.05))
    #     plt.grid(color='b', linestyle="-", linewidth=0.05)
    #     plt.xlabel("f")
    #     plt.ylabel("r(f)")
    #     plt.title("centrality_metric")

    # uncomment this for q13 - comparing R, degree and strength to R'
    #     centrality_metric = []
    #     for f in np.linspace(0.05, 0.5, 10):
    #         size_f = int(f * flat_set.N)
    #         degF = len(set(ranking_avg[0:size_f]).intersection(set(ranking_degree[0:size_f]))) / size_f
    #         strF = len(set(ranking_avg[0:size_f]).intersection(set(ranking_strenght[0:size_f]))) / size_f
    #         rF = len(set(ranking_avg[0:size_f]).intersection(set(ranking_infect[0:size_f]))) / size_f
    #         centrality_metric.append((degF, strF, rF))
    #
    #     p1, p2, p3 = plt.plot(centrality_metric)
    #     plt.legend([p1, p2, p3], ["Degree", "Strength", "R"])
    #     y = np.array(centrality_metric)
    #     plt.xticks(np.arange(10), [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    #     plt.grid(color='b', linestyle="-", linewidth=0.05)
    #     plt.xlabel("f")
    #     plt.ylabel("r(f)")
    #     plt.title("Centrality features")

    # uncomment this to plot infection rate E[I(t)]
    # plt.plot(range(0, len(infect_num_expect)), infect_num_expect, 'k', color='#3F7F4C')
    # plt.fill_between(range(0, len(infect_num_expect)), infect_num_expect_min, infect_num_expect_plus,
    #                 alpha=1, edgecolor='#3F7F4C', facecolor='#7EFF99',
    #                 linewidth=0)
    # plt.ylabel('avg num of infected nodes & error bar')
    # plt.xlabel('timestep')
    # plt.title("Average number of infected nodes & its error bar")

    plt.show()
