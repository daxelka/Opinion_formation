import networkx as nx
import time
from ABM.deffuant import DeffuantModel
from distribution_tools import uniform_opinion

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

# Clusters in final opinion
clusters, means = model.clusters_detector(model.get_opinion())
print('Means of clusters:', means)

# Show opinion distribution
model.show_opinion_distribution(model.get_opinion())

sigmas = [0.5]
for sigma in sigmas:  # iteration through the family of initial opinions
    for i in range(1):  # iteration through realisations of each distribution
        initial_opinion = uniform_opinion(N_nodes, 0.5, sigma)
        # Set initial opinion
        model.set_opinion(initial_opinion)
        # Run the model
        model.opinion_formation()
        # Get final opinion
        model.show_opinion_distribution(model.get_opinion())