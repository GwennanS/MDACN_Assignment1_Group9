import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import networkx


class Link(object):

    def __init__(self, a, b):
        self.node1 = a
        self.node2 = b
        timeStamps = []

    def add_time(self, t):
        self.timeStamps.append(t)

    def __repr__(self):
        return "(a: {0}, b: {1}, t: {2})".format(self.node1, self.node2, self.timeStamps)


class FlatSet(object):
    def __init__(self, flat_set):
        self.network = flat_set
        self.links = []
        self.degrees = []
        self.degrees_dict = dict
        self.dev = 0
        self.set_links()
        self.set_standard_deviation_degree()

    def set_links(self):
        for index, row in self.network.iterrows():
            if self.links.count((row.node1, row.node2)) == 0:
                self.links.append((row.node1, row.node2))

    def set_standard_deviation_degree(self):
        sum = 0
        for n in self.get_nodes():
            degree = len([item for item in self.links if n in item])
            self.degrees.append((n, degree))
            sum += (degree - self.get_average_degree()) ** 2
        self.dev = (sum / (self.getN() - 1)) ** 0.5
        self.degrees_dict = dict(self.degrees)

    def get_nodes(self):
        uniques = np.unique(self.network[['node1', 'node2']].values)
        return uniques

    def getN(self):
        return self.get_nodes().size

#    def get_links(self):
#        return self.links

    def getL(self):
        return len(self.links)

    def get_average_degree(self):
        return (2 * self.getL() / self.getN())

#    def get_standard_deviation_degree(self):
#        return self.dev

    def assortativity (self):
        sumEDD = 0
        sumEDmin = 0
        sumEDmplus = 0
        varD = self.dev
        for l in self.links:
            sumEDD += self.degrees_dict[l[0]]*self.degrees_dict[l[1]]
            sumEDmin += self.degrees_dict[l[0]]
            sumEDmplus += self.degrees_dict[l[1]]
        EDD = sumEDD / self.getL()
        EDmin = sumEDmin / self.getL()
        EDplus = sumEDmplus / self.getL()
        return (EDD - EDplus*EDmin)/varD

    def clustering_coefficient(self):
        sumccn = 0
        for n in self.get_nodes():
            deg = self.degrees_dict[n]
            if deg == 1:
                ccn = 0
            else:
                links = [item for item in self.links if n in item]
                temp = set((j for i in links for j in i))
                temp.discard(n)
                num_links = self.count_links(temp)
                ccn = num_links/((deg*(deg-1))/2)
            sumccn += ccn
        return sumccn/self.getN()

    def count_links(self, nodes):
        count = 0
        for node1 in nodes:
            for node2 in nodes:
                if self.links.count((node1, node2)) == 1:
                    count = count+1
        return count




def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('Complex Networks')
    df = pd.read_excel('manufacturing_emails_temporal_network.xlsx')
    # flat_set = df.values.tolist()
    class_set = FlatSet(df)
    # link_set = []
    # for p in flat_set:
    #    link_set.append(Link(*p))
    # print(df[0:5])
    print("N: ", class_set.getN())
    print("L: ", class_set.getL())
    print("average degree: ", class_set.get_average_degree())
    print("standard deviation of the degree: ", class_set.dev)
    #print("degree correlation (assortativity): ", class_set.assortativity())
    #print("clustering coefficient: ", class_set.clustering_coefficient())

    plt.plot(class_set.degrees)
    plt.show()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
