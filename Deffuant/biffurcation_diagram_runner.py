import networkx as nx
import numpy as np
import time
from utils.bifurcation_diagram.generator import BifurcationDiagramGenerator
from utils.bifurcation_diagram.plotter import BifurcationDiagramPlotter
from utils.libs import drange
from deffuant import DeffuantModel
from distribution_tools import uniform_opinion

def parameter_iterator():
    return drange(0.06, 0.51, 0.01)

def initial_values_iterator():
    initial_opinions = []
    for i in range(10):
        distribution = uniform_opinion(N_nodes)
        initial_opinions.append(distribution)
    return initial_opinions


def run(parameter, initial_value):
    model = DeffuantModel(G, parameter, 0.5)
    model.set_opinion(initial_value)
    model.opinion_formation()
    clusters, means = model.clusters_detector(model.get_opinion())
    # Filter by cluster density
    densities = model.cluster_density(clusters)
    major_groups = np.array(means)[np.array(densities) > 0.1]
    # print("means:", means, "densities:", densities, "major", major_groups)
    return major_groups

# Graph Inisialisation
N_nodes: int = 100
G = nx.complete_graph(N_nodes)

t0 = time.time()
generator = BifurcationDiagramGenerator(parameter_iterator, initial_values_iterator, run)

x_var, y_var = generator.run()
t1 = time.time()
print("performance time", t1-t0)

BifurcationDiagramPlotter().plot(x_var, y_var, 'confidence bound', 'opinion')

