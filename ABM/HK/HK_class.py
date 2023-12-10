import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random


class HKModel:
    def __init__(self, G, confidence_interval, cautiousness):
        # graph parameters
        self.G = G
        self.N_nodes = len(G)
        self.edges = list(self.G.edges(data=False))
        self.nodes = list(self.G.nodes(data=False))
        # Deffuant parameters
        self.confidence = confidence_interval  # restricted to interval (0, 0.5]
        self.cautiousness = cautiousness  # convergence parameter, restricted to interval (0, 0.5]
        self.PRECISION = 0.01  # difference between opinions that are considered identical
        # clusters parameters
        self.CLUSTER_PRECISION = 0.01  # difference in neighbouring opinions belonging to the same cluster
        self.CLUSTER_MAX_LENGTH = 0.1
        self.MINIMUM_DIFFERENCE = 0.01  # difference in opinion updating after which we consider that the opinion does not change
        # convergence parameters
        self.MAXIMUM_STEPS = int(5e06)
        self.IDLE_STEPS = 100

    def interaction(self):
        node = random.choice(self.nodes)
        # neighbors_within_radius = [n for n in self.G.neighbors(node) if
        #                            abs(self.G.nodes[node]['opinion'] - self.G.nodes[n]['opinion']) <= self.confidence]
        # selected_nodes = neighbors_within_radius + [node]
        # avg_opinion = sum(self.G.nodes[n]['opinion'] for n in selected_nodes) / len(selected_nodes)

        node_opinion = self.G.nodes[node]['opinion']
        opinions_neighbours_within_confidence = [self.G.nodes[n]['opinion'] for n in self.G.neighbors(node) if
                                                abs(node_opinion- self.G.nodes[n]['opinion']) <= self.confidence]
        selected_opinions = opinions_neighbours_within_confidence + [node_opinion]
        avg_opinion = sum(selected_opinions ) / len(selected_opinions)

        diff = np.abs(node_opinion - avg_opinion)
        self.G.nodes[node]['opinion'] = avg_opinion
        return diff


    def one_step(self):
        self.interaction()

    def run(self,n_steps):
        for _ in range(n_steps):
            self.one_step()

    def opinion_formation(self):
        n_idle_steps = 0
        total_steps = 0
        not_convergence = True
        idle_continuous_steps = True

        while not_convergence:
            diff = self.interaction()
            if diff < self.MINIMUM_DIFFERENCE:
                if idle_continuous_steps:  # check if the previous steps were idle too
                    n_idle_steps += 1
                else:
                    idle_continuous_steps = True
                    n_idle_steps = 0
            else:
                idle_continuous_steps = False

            total_steps += 1
            not_convergence, n_idle_steps = self.is_not_convergence(n_idle_steps, total_steps)
        return total_steps

    def is_not_convergence(self, n_idle_steps, total_steps):
        if total_steps < self.MAXIMUM_STEPS:
            if n_idle_steps >= self.IDLE_STEPS:
                _, means = self.clusters_detector(self.get_opinion())
                if np.all(np.diff(means) > self.confidence):
                    # print('model has converged, steps performed:', str(total_steps))
                    # print('Means of clusters:', means)
                    return False, n_idle_steps  # model converged, opinion formation stops
                else:
                    return True, 0  # not converged, restart idle steps counter
            else:
                return True, n_idle_steps  # not converged, keep counting idle steps
        else:
            print('model not converging, maximum steps performed:', str(total_steps))
            return False, n_idle_steps  # not converged, opinion formation stops

    def clusters_detector(self, opinion):
        clusters = []
        means = []
        points_sorted = sorted(opinion)
        curr_point = points_sorted[0]
        curr_cluster = [curr_point]
        for point in points_sorted[1:]:
            if point <= curr_point + self.CLUSTER_PRECISION:
                curr_cluster.append(point)
            else:
                # append new cluster to clusters
                clusters.append(curr_cluster)
                means.append(np.mean(curr_cluster))
                # start new cluster
                curr_cluster = [point]
            curr_point = point
        clusters.append(curr_cluster)
        means.append(np.mean(curr_cluster))
        return clusters, means

    def cluster_density(self, clusters):
        density = []
        for cluster in clusters:
            density.append(len(cluster)/self.N_nodes)
        return density


    def show_opinion_distribution(self, opinions):
        bins = [0.01 * n for n in range(100)]
        plt.hist(opinions, bins=bins, density=True)
        plt.title("Histogram of opinions")
        plt.show()

    def set_opinion(self, opinion_array):
        d = dict(enumerate(opinion_array))  # transforming numpy.array into dictionary
        nx.set_node_attributes(self.G, d, 'opinion')

    def get_opinion(self):
        opinions = np.array(list(nx.get_node_attributes(self.G, 'opinion').values()))
        return opinions