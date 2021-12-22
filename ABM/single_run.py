import networkx as nx
import time
from ABM.deffuant import DeffuantModel
from distribution_tools import inverse_transform_sampling
import numpy as np

# Initiating a graph
N_nodes: int = 100
G = nx.complete_graph(N_nodes)
# G = nx.empty_graph(N_nodes)

# Initiating a Deffuant model on the graph
t0 = time.perf_counter()
model = DeffuantModel(G, 0.5, 0.5)

# Setting initial opinion
# initial_opinion = normal_opinion(N_nodes, 0.1, 1, 0, 1)
# initial_opinion = uniform_opinion(N_nodes)
def gen_pdf(n_peaks, epsilon):
    def pdf(x):
        f = np.ones(x.shape) + epsilon * np.cos((2*n_peaks-1)*np.pi*x)
        f / np.trapz(f, x)
        return f
    return pdf

pdf = gen_pdf(3, 0.1)
initial_opinion = inverse_transform_sampling(pdf, N_nodes, (0, 1))
model.show_opinion_distribution(initial_opinion)

# Set initial opinion
model.set_opinion(initial_opinion)

# Run the model
model.opinion_formation()
t1 = time.perf_counter()
print('performance time:', t1 - t0)

clusters, means = model.clusters_detector(model.get_opinion())
densities = model.cluster_density(clusters)
print('menas:', means)
print('densities:',densities)


# Clusters in final opinion
clusters, means = model.clusters_detector(model.get_opinion())
print('Means of clusters:', means)

# Show opinion distribution
model.show_opinion_distribution(model.get_opinion())