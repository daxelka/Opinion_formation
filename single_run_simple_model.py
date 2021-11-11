import networkx as nx
import time
from deffuant_simple import DeffuantModelSimple
from distribution_tools import normal_opinion
from distribution_tools import uniform_opinion
from distribution_tools import inverse_transform_sampling
import numpy as np

# Initiating a opinions
N_nodes: int = 100

def gen_pdf(n_peaks, epsilon):
    def pdf(x):
        f = np.ones(x.shape) + epsilon * np.cos((2*n_peaks-1)*np.pi*x)
        f / np.trapz(f, x)
        return f
    return pdf


t0 = time.perf_counter()
pdf = gen_pdf(3, 0.1)

initial_opinion = inverse_transform_sampling(pdf, N_nodes, (0, 1))

# Initiate the model
model = DeffuantModelSimple(N_nodes, 0.2, 0.5)

# Set initial conditions
model.set_opinion(initial_opinion)

model.show_opinion_distribution(initial_opinion)

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