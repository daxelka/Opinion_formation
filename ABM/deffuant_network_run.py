import networkx as nx
import time
from ABM.deffuant_network_class import DeffuantModelNetwork
import distribution_tools as tools
from distribution_tools import inverse_transform_sampling
import numpy as np

# Initiating a graph
N_nodes: int = 100
G = nx.complete_graph(N_nodes)

# Initiating a Deffuant model on the graph
model = DeffuantModelNetwork(G, 0.5, 0.5)

# Setting initial opinion
initial_opinion = tools.uniform_opinion(N_nodes)
model.show_opinion_distribution(initial_opinion)
model.set_opinion(initial_opinion)

# Run the model
final_opinion = model.opinion_formation()
model.show_opinion_distribution(final_opinion)
clusters, means = model.clusters_detector(final_opinion)
print('Means of clusters:', means)
