import networkx as nx
import math
from pathos.multiprocessing import ProcessingPool as Pool
import time
import json

from simulation import Simulation
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
N_nodes: int = 2000
# Number of run for each parameter
n_runs = 10

# Create a set of initial distributions
initial_opinions = []
for sigma in range(n_runs):
    distribution = uniform_opinion(N_nodes)
    initial_opinions.append(distribution)
# Create a list of n_runs for different intervals
ic_range = [10, 10, 10, 10]

# Create a set of parameter intervals
intervals = [(1, 2), (2, 4), (4, 5), (5, 6)]
step = 0.05

parameter_range = []
for interval in intervals:
    start, end = interval
    size = math.ceil((end - start) / step)
    deltas = np.linspace(start, end, num=size, endpoint=False)
    epsilons = 0.5 / deltas
    parameter_range.append(epsilons)


def one_iteration(N_nodes, initial_opinions, parameter_range, ic_range, index):
    from simulation import Simulation
    from deffuant_simple import DeffuantModelSimple

    def run(parameter, initial_value):
        # Initiate a model with confidence bound specified in parameter
        confidence_bound, cautiousness = parameter, 0.5
        model = DeffuantModelSimple(N_nodes, confidence_bound, cautiousness)
        # Set initial condition
        model.set_opinion(initial_value)
        # Run opinion formation on the model
        model.opinion_formation()
        # if model converged to return opinion if not to return []
        if model.get_opinion():
            # Identify clusters and their means
            clusters, means = model.clusters_detector(model.get_opinion())
            # Calculate clusters densities
            densities = model.cluster_density(clusters)
            # Add results to data
            result = []
            for count, mean_value in enumerate(means):
                density = densities[count]
                result.append([mean_value, density])

        else:
            return []

        return result

    def initial_values_iterator():
        n_ic = ic_range[index]
        return initial_opinions[:n_ic]

    # we split work between processes by parameter
    # index is a process specific part of the job
    def parameter_iterator():
        return parameter_range[index]

    generator = Simulation(parameter_iterator, initial_values_iterator, run)

    return generator.run()


def unpack(lists):
    merged_lists = {**lists[0], **lists[1], **lists[2], **lists[3]}
    return merged_lists


total_cores = 4
p = Pool(total_cores)
t0 = time.time()

f = partial(one_iteration, N_nodes, initial_opinions, parameter_range, ic_range)

try:
    experiments = unpack(p.map(f, range(total_cores)))
finally:
    p.terminate()

t1 = time.time()

print("performance time", t1 - t0)

# Writing to json
data = {'setup': {'N_nodes': N_nodes,
                  'step': step,
                  'notes': 'uniform IC plus cos disturbance with 2 negative peaks and amplitude 0.5 '},
        'experiments': experiments,
        'initial_conditions': [r.tolist() for r in initial_opinions]
        }

filename = '/Users/daxelka/Research/Deffuant_model/ABM_simulation/data/test.txt'

with open(filename, 'w') as outfile:
    json.dump(data, outfile)
print('json created')