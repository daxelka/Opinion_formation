import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
from time import perf_counter


class DeffuantModel:
    def __init__(self, G, confidence_interval, cautiousness):
        # graph parameters
        self.G = G
        self.edges = list(self.G.edges(data=False))
        # Deffuant parameters
        self.confidence = confidence_interval  # restricted to interval (0, 0.5]
        self.cautiousness = cautiousness  # convergence parameter, restricted to interval (0, 0.5]
        self.PRECISION = 0.01   # difference between opinions that are considered identical
        # clusters parameters
        self.CLUSTER_PRECISION = 0.05  # difference in neighbouring opinions belonging to the same cluster
        self.CLUSTER_MAX_LENGTH = 0.2
        # convergence parameters
        self.MAXIMUM_STEPS = 100000
        self.STEPS_MONITORED = 100
        self.IDLE_STEPS = 90

    def formation(self, node1, node2):
        """
        value = (node1.value + node2.value) * self.persuasibility
        """
        value1 = self.G.nodes[node1]['opinion']
        value2 = self.G.nodes[node2]['opinion']
        diff = abs(value1 - value2)
        if diff < self.confidence and diff > self.PRECISION:
            value = (value1 + value2) * self.cautiousness
            return value
        elif diff < self.PRECISION:
            return 0
        else:
            return False

    def opinion_formation(self):
        counter = 0
        loop = 1
        total_steps = 0
        not_convergence = True
        while not_convergence:
            # G.edges returns a tuple with two node numbers
            edge = random.choice(self.edges)
            # random edges
            value = self.formation(*edge)
            if value > 0:
                node1, node2 = edge
                self.G.nodes[node1]['opinion'] = value
                self.G.nodes[node2]['opinion'] = value
            elif value == 0:
                counter += 1
            # else:
            #     # remove edge
            #     self.G.remove_edge(*edge)
            total_steps += 1
            loop, counter, not_convergence = self.is_not_convergence(loop, counter, total_steps)
        return self.G

    def is_not_convergence(self, loop, counter, total_steps):
        if total_steps < self.MAXIMUM_STEPS:
            if loop == self.STEPS_MONITORED:
                if counter >= self.IDLE_STEPS:
                    print('model has converged')
                    print('steps performed:', str(total_steps))
                    return 1, 1, False
                else:
                    return 1, 0, True
            else:
                loop += 1
                return loop, counter, True
        else:
            print('model is not converging')
            return loop, counter, False

    def clusters_detector(self):
        clusters = []
        points_sorted = sorted(self.get_opinion())
        curr_point = points_sorted[0]
        curr_cluster = [curr_point]
        for point in points_sorted[1:]:
            if point <= curr_point + self.CLUSTER_PRECISION:
                curr_cluster.append(point)
            else:
                clusters.append(curr_cluster)
                curr_cluster = [point]
            curr_point = point
        clusters.append(curr_cluster)
        return clusters


    def show_opinion_distribution(self, opinions):
        # opinions = np.array(list(nx.get_node_attributes(self.G, 'opinion').values()))
        # hist_data = np.histogram(opinions.values())
        bins = [0.01 * n for n in range(100)]
        plt.hist(opinions, bins=bins)
        plt.title("Histogram of opinions")
        plt.show()
        # plt.savefig('result2.png')

    def set_opinion(self, opinion_array):
        # transforming numpy.array into dictionary
        d = dict(enumerate(opinion_array))
        nx.set_node_attributes(G, d, 'opinion')

    def get_opinion(self):
        opinions = np.array(list(nx.get_node_attributes(self.G, 'opinion').values()))
        return opinions


# Initiating a graph
N_nodes: int = 100
G = nx.complete_graph(N_nodes)
# G = nx.empty_graph(N_nodes)

# Initiating a model on the graph
model = DeffuantModel(G, 0.2, 0.5)


def random_opinion(N_nodes):
    rng = np.random.default_rng()
    opinion_distribution = rng.random((N_nodes,))
    return opinion_distribution


def uniform_opinion(N_nodes):
    rng = np.random.default_rng()
    opinion = rng.uniform(0.0, 1.0, (N_nodes,))
    return opinion

def sin_opinion(N_nodes):
    # randomly chosen N_nodes numbers from [0,1) from uniform distribution
    rng = np.random.default_rng()
    values = rng.uniform(0.0, 1.0, (N_nodes,))
    m = 1
    # opinion_distribution = values + m * np.sin(2*math.pi/k * np.linspace(0, 1, N_nodes, endpoint=False))
    opinion = m * np.arccos(np.linspace(-1, 1, N_nodes))/math.pi
    return opinion

def normal_opinion(N_nodes, mu, sigma):
    rng = np.random.default_rng()
    opinion = rng.normal(mu, sigma, N_nodes)
    return opinion

def multimodal_normal_opinion(N_nodes, sigma):
    rng = np.random.default_rng()
    mus = [.25, .75]
    opinion = np.zeros(N_nodes)
    for mu in mus:
        opinion += rng.normal(mu, sigma, N_nodes)
    return opinion


# Setting initial opinion
initial_opinion = uniform_opinion(N_nodes)
# initial_opinion = normal_opinion(N_nodes, 0.5, 0.1)
model.show_opinion_distribution(initial_opinion)

# Set initial opinion
model.set_opinion(initial_opinion)
# Run the model
model.opinion_formation()
print('Number of clusters:', len(model.clusters_detector()))
# Get final opinion
model.show_opinion_distribution(model.get_opinion())

# sigmas = [0.5]
# for sigma in sigmas:  # iteration through the family of initial opinions
#     for i in range(1):  # iteration through realisations of opinion distribution
#         initial_opinion = normal_opinion(N_nodes, 0.5, sigma)
#         initial_opinion = multimodal_normal_opinion(N_nodes, sigma, 2)
#         # Set initial opinion
#         model.set_opinion(initial_opinion)
#         # Run the model
#         model.opinion_formation()
#         # Get final opinion
#         model.show_opinion_distribution(model.get_opinion())



