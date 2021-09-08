import networkx as nx
import numpy as np
import time
import pickle
from utils.bifurcation_diagram.generator import BifurcationDiagramGenerator
from utils.bifurcation_diagram.plotter import BifurcationDiagramPlotter
from utils.libs import drange
from deffuant import DeffuantModel
from distribution_tools import normal_opinion
from distribution_tools import uniform_opinion


def parameter_iterator():
    return drange(0.1, 0.5, 0.01)

def initial_values_iterator():
    initial_opinions = []
    # for sigma in drange(0.1, 0.9, 0.1):
    #     for mu in drange(0.3, 0.7, 0.1):
    #         # distribution = uniform_opinion(N_nodes)
    #         distribution = normal_opinion(N_nodes, mu, sigma, 0, 1)
    #         initial_opinions.append(distribution)
    # for sigma in drange(0.1, 0.9, 0.05):
    #         distribution = normal_opinion(N_nodes, 0.5, sigma, 0, 1)
    #         initial_opinions.append(distribution)
    for sigma in range(10):
            distribution = uniform_opinion(N_nodes)
            initial_opinions.append(distribution)

    return initial_opinions


def run(parameter, initial_value):
    confidence_bound, cautiousness = parameter, 0.5
    model = DeffuantModel(G,confidence_bound, cautiousness)
    model.set_opinion(initial_value)
    model.opinion_formation()
    clusters, means = model.clusters_detector(model.get_opinion())
    # Filter by cluster density
    densities = model.cluster_density(clusters)
    major_groups = np.array(means)[np.array(densities) > 0.1]
    # print("means:", means, "densities:", densities, "major", major_groups)
    return major_groups

# Graph Inisialisation
N_nodes: int = 1000
G = nx.complete_graph(N_nodes)

t0 = time.time()
generator = BifurcationDiagramGenerator(parameter_iterator, initial_values_iterator, run)

x_var, y_var = generator.run()
t1 = time.time()
print("performance time", t1-t0)

BifurcationDiagramPlotter().plot(x_var, y_var, 'confidence bound', 'opinion')

