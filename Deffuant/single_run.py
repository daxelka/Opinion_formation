import networkx as nx
import time
from deffuant import DeffuantModel
from distribution_tools import uniform_opinion
import numpy as np

# Initiating a graph
N_nodes: int = 1000
G = nx.complete_graph(N_nodes)
# G = nx.empty_graph(N_nodes)

# Initiating a Deffuant model on the graph
model = DeffuantModel(G, 0.2, 0.5)

t0 = time.time()
# Setting initial opinion
initial_opinion = uniform_opinion(N_nodes)
model.show_opinion_distribution(initial_opinion)

# Set initial opinion
model.set_opinion(initial_opinion)

# Run the model
model.opinion_formation()
t1 = time.time()
print('performance time:', t1 - t0)

clusters, means = model.clusters_detector(model.get_opinion())
densities = model.cluster_density(clusters)
print(densities)
np.array(means)[np.array(densities)>0.1]
# list(i >= 0.1 for i in model.cluster_density(clusters))

# # Clusters in final opinion
# clusters, means = model.clusters_detector(model.get_opinion())
# print('Means of clusters:', means)

# Show opinion distribution
model.show_opinion_distribution(model.get_opinion())