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
        self.PRECISION = 0.01
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
            else:
                # remove edge
                self.G.remove_edge(*edge)
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

    def show_opinion_distribution(self):
        opinions = np.array(list(nx.get_node_attributes(self.G, 'opinion').values()))
        # hist_data = np.histogram(opinions.values())
        bins = [0.01 * n for n in range(100)]
        plt.hist(opinions, bins=bins)
        plt.title("Histogram of opinions")
        plt.show()
        # plt.savefig('result2.png')

    def set_initial_opinion(self, opinion_array):
        # transforming numpy.array into dictionary
        d = dict(enumerate(opinion_array))
        nx.set_node_attributes(G, d, 'opinion')


# Initiating a graph
t_start = perf_counter()
N_nodes: int = 1000
G = nx.complete_graph(N_nodes)
t_stop = perf_counter()  # stop stopwatch
print("Graph creation time:", t_stop - t_start, 's')

# Initiating a model on the graph
t_start = perf_counter()
model = DeffuantModel(G, 0.2, 0.5)
t_stop = perf_counter()  # stop stopwatch
print("Model initialisation time:", t_stop - t_start, 's')

# Setting initial opinion
rng = np.random.default_rng()

# randomly chosen N_nodes numbers from [0,1) from uniform distribution
t_start = perf_counter()
opinion_distribution = rng.random((N_nodes,))
model.set_initial_opinion(opinion_distribution)
t_stop = perf_counter()  # stop stopwatch
print("Initial opinion population time:", t_stop - t_start, 's')

# Running the model
t_start = perf_counter()  # start stopwatch
model.opinion_formation()
t_stop = perf_counter()  # stop stopwatch
print("Opinion formation time:", t_stop - t_start, 's')

# Plotting the results
# print(G.nodes(data=True))
model.show_opinion_distribution()

