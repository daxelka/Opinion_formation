import math
import matplotlib.pyplot as plt
import numpy as np
import random
from utils import libs


class DeffuantModelPolar:
    def __init__(self, N_nodes, confidence_interval, cautiousness, jump_radius, jump_frequency):
        self.N_nodes = N_nodes
        self.x_step = 1/N_nodes * 2 * math.pi
        self.opinions = []
        # Deffuant parameters
        self.confidence = confidence_interval * 2 * math.pi  # restricted to interval (0, 0.5]
        self.cautiousness = 0.5  # convergence parameter, restricted to interval (0, 0.5]
        self.PRECISION = 0.01 * 2 * math.pi  # difference between opinions that are considered identical
        # clusters parameters
        self.CLUSTER_PRECISION = 0.02 * 2 * math.pi  # difference in neighbouring opinions belonging to the same cluster
        self.CLUSTER_MAX_LENGTH = 0.1 * 2 * math.pi
        # convergence parameters
        self.MAXIMUM_STEPS = 10000000
        self.IDLE_STEPS = 100
        self.node_ids = range(self.N_nodes)
        self.converged = None
        self.jump_radius = jump_radius * 2 * math.pi
        self.jump_frequency = jump_frequency * 100
        self.rng = np.random.default_rng()

    def closest_average_angle(self, angle1, angle2):
        diff1 = abs(angle1 - angle2)
        diff2 = 2 * math.pi - abs(angle1 - angle2)
        if diff1 <= diff2:
            return min(angle1, angle2) + diff1 / 2
        else:
            if max(angle1, angle2) + diff2 / 2 < 2 * math.pi:
                return max(angle1, angle2) + diff2 / 2
            else:
                return max(angle1, angle2) + diff2 / 2 - 2 * math.pi

    def interaction(self):
        # choosing two nodes for interaction at random
        node1, node2 = random.sample(self.node_ids, 2)
        value1 = self.opinions[node1]
        value2 = self.opinions[node2]
        diff = min(abs(value1 - value2), 2 * math.pi - abs(value1 - value2))
        if diff < self.confidence and diff > self.PRECISION:
            self.opinions[node1] = self.closest_average_angle(value1, value2)
            self.opinions[node2] = self.closest_average_angle(value1, value2)
            return diff
        elif diff < self.PRECISION:
            return 0
        else:
            return False

    def random_jump(self):
        node = random.sample(self.node_ids, 1)[0]
        value = self.opinions[node]
        new_value = list(self.rng.uniform(value - self.jump_radius,
                                           value + self.jump_radius,
                                           (1,)))[0]
        if new_value > 2 * math.pi:
            new_value = new_value - 2 * math.pi
        elif new_value < 0:
            new_value = new_value + 2 * math.pi
        self.opinions[node] = new_value

    def single_step(self):
        # take a probability of random jump
        m = random.randint(1, 100)

        if m <= self.jump_frequency:
            self.random_jump()

        else:
            self.interaction()

    def one_step(self):
        self.interaction()
        return self.opinions

    def opinion_formation(self, until_converged=False, n_steps=100):

        if until_converged:
            n_idle_steps = 0
            total_steps = 0
            not_convergence = True
            idle_continuous_steps = True

            while not_convergence:
                value = self.interaction()
                # value = self.single_step()
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
        else:
            for i in range(n_steps):
                self.single_step()

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
        bins = [0.01 * n for n in range(628)]
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

