import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random


class DeffuantBarabasiModel:
    def __init__(self, N_nodes, confidence_interval, cautiousness):
        self.N_nodes = N_nodes
        self.opinions = []
        # Deffuant parameters
        self.confidence = confidence_interval  # restricted to interval (0, 0.5]
        self.cautiousness = cautiousness  # convergence parameter, restricted to interval (0, 0.5]
        self.PRECISION = 0.01  # difference between opinions that are considered identical
        # clusters parameters
        self.CLUSTER_PRECISION = 0.08  # difference in neighbouring opinions belonging to the same cluster
        self.CLUSTER_MAX_LENGTH = 0.1
        # convergence parameters
        self.MAXIMUM_STEPS = int(1e06)
        self.IDLE_STEPS = 100
        self.node_ids = range(self.N_nodes)
        self.converged = None
        self.rng = np.random.default_rng()

    def dw_interaction(self, node1):
        # choosing a second node for interaction from numeric opinions
        node2 = random.choice(self.node_ids)
        value1 = self.opinions[node1]
        value2 = self.opinions[node2]

        if not np.isnan(value2):
            diff = abs(value1 - value2)
            if diff < self.confidence and diff > self.PRECISION:
                self.opinions[node1] = value1 + self.cautiousness * (value2 - value1)
                self.opinions[node2] = value2 + self.cautiousness * (value1 - value2)
                return diff
            elif diff < self.PRECISION:
                return 0
            else:
                return -999
        else:
            return -999

    def preferential_adoption(self, node1):
        numeric_opinions = [x for x in self.opinions if np.isfinite(x)]
        hist, edges = np.histogram(numeric_opinions, bins=100, range=(0, 1))
        p = random.randint(0, len(numeric_opinions))

        sum_hist = 0
        for index, value in enumerate(hist):
            sum_hist += value
            if sum_hist >= p:
                new_opinion = (edges[index] + edges[index + 1]) / 2
                self.opinions[node1] = new_opinion
                break
        return -999

    def single_step(self):
        # pick node at random
        node1 = random.choice(self.node_ids)
        value1 = self.opinions[node1]
        # check if the node have a non NA opinion
        if np.isnan(value1):
            # print('B: value1 ' + str(value1))
            return_value = self.preferential_adoption(node1)
        else:
            # print('DW: value1 ' + str(value1))
            return_value = self.dw_interaction(node1)
        return return_value

    def run(self, n_steps):
        for i in range(n_steps):
            self.single_step()


    def opinion_formation(self):
        # needs to be corrected, def.interaction is changed and called dw_interaction
        n_idle_steps = 0
        total_steps = 0
        not_convergence = True
        idle_continuous_steps = True

        while not_convergence:
            value = self.single_step() #here I will need to change to single step and do something about returned value
            if value > 0:
                idle_continuous_steps = False
            elif value == 0:
                if idle_continuous_steps:  # check if the previous steps were idle too
                    n_idle_steps += 1
                else:
                    idle_continuous_steps = True
                    n_idle_steps = 0

            total_steps += 1
            not_convergence, n_idle_steps = self.is_not_convergence(n_idle_steps, total_steps)
        return self.opinions

    def is_not_convergence(self, n_idle_steps, total_steps):
        if total_steps < self.MAXIMUM_STEPS:
            if n_idle_steps >= self.IDLE_STEPS:
                _, means = self.clusters_detector(self.opinions)
                if np.all(np.diff(means) > self.confidence):
                    self.converged = True
                    return False, n_idle_steps  # model converged, opinion formation stops
                else:
                    return True, 0  # not converged, restart idle steps counter
            else:
                return True, n_idle_steps  # not converged, keep counting idle steps
        else:
            print('model not converging, maximum steps performed:', str(total_steps))
            self.converged = False
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
        self.opinions = list(opinion_array)

    def get_opinion(self):
        if self.converged:
            opinions = self.opinions
        else:
            opinions = []
        return opinions

    def get_unconverged_opinion(self):
        return self.opinions

