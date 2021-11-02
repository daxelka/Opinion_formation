import networkx as nx
from pathos.multiprocessing import ProcessingPool as Pool
import time

from utils.bifurcation_diagram.plotter import BifurcationDiagramPlotter
import numpy as np
from distribution_tools import uniform_opinion
from functools import partial


# def gen_pdf(n_peaks, epsilon):
#     def pdf(x):
#         f = np.ones(x.shape) + epsilon * np.cos((2*n_peaks)*np.pi*x)
#         f / np.trapz(f, x)
#         return f
#     return pdf

# Graph Initialisation
N_nodes: int = 10000

# Create a set of initial distributions
initial_opinions = []
for sigma in range(1):
    distribution = uniform_opinion(N_nodes)
    initial_opinions.append(distribution)


def one_iteration(total_cores, N_nodes, initial_opinions, index):
    from utils.bifurcation_diagram.generator import BifurcationDiagramGenerator
    from utils.libs import drange
    from deffuant_simple import DeffuantModelSimple
    import numpy as np

    def run(parameter, initial_value):
        confidence_bound, cautiousness = parameter, 0.5
        model = DeffuantModelSimple(N_nodes, confidence_bound, cautiousness)
        model.set_opinion(initial_value)
        model.opinion_formation()
        clusters, means = model.clusters_detector(model.get_opinion())
        # Filter by cluster density
        densities = model.cluster_density(clusters)
        major_groups = np.array(means)[np.array(densities) > 0.1]
        return major_groups

    def initial_values_iterator():
        return initial_opinions

    # we split work between processes by parameter
    # index is a process specific part of the job
    def parameter_iterator():
        start = 0.1 + (0.5 - 0.1) / total_cores * index
        end = 0.1 + (0.5 - 0.1) / total_cores * (index + 1)

        return drange(start, end, 0.01)

    generator = BifurcationDiagramGenerator(parameter_iterator, initial_values_iterator, run)

    return generator.run()


def unpack(lists):
    xs = []
    ys = []

    for l in lists:
        xs.extend(l[0])
        ys.extend(l[1])

    return [xs, ys]


total_cores = 4
p = Pool(total_cores)
t0 = time.time()

f = partial(one_iteration, total_cores, N_nodes, initial_opinions)

try:
    x_var, y_var = unpack(p.map(f, range(total_cores)))
finally:
    p.terminate()

t1 = time.time()

print("performance time", t1 - t0)

# transformation to (y, delta)
y = (np.array(y_var) - 0.5) / np.array(x_var)
x = 0.5 / np.array(x_var)

BifurcationDiagramPlotter().plot(x_var, y_var, 'confidence bound', 'opinion')
