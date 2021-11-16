import networkx as nx
import math
from pathos.multiprocessing import ProcessingPool as Pool
import time
import json

from simulation import Simulation
import numpy as np
from distribution_tools import uniform_opinion
from distribution_tools import inverse_transform_sampling
from distribution_tools import show_distribution
from functools import partial


def gen_pdf(n_peaks, epsilon):
    def pdf(x):
        f = np.ones(x.shape) + epsilon * np.cos((2*n_peaks)*np.pi*x)
        f / np.trapz(f, x)
        return f
    return pdf

# Graph Initialisation
N_nodes: int = 10000
# Number of run for each parameter
n_runs = 10
n_repetitions = 5
# Create a set of initial distributions
initial_opinions = []


for sigma in range(n_runs):
    distribution = uniform_opinion(N_nodes)
    for m in range(n_repetitions):
        initial_opinions.append(distribution)

show_distribution(initial_opinions[0])

# Create a set of parameter intervals
# intervals = [(1, 1.5), (1.5, 1.75), (1.75, 2), (2, 2.25), (2.25, 2.5),
#              (2.5, 2.75), (2.75, 3), (3, 3.25), (3.25, 3.5), (3.5, 3.75),
#              (3.75, 4)]

# step = 0.05
#
# parameter_range = []
# for interval in intervals:
#     start, end = interval
#     size = math.ceil((end - start) / step)
#     deltas = np.linspace(start, end, num=size, endpoint=False)
#     epsilons = 0.5 / deltas
#     parameter_range.append(epsilons)

def get_intervals(start, end, n_intervals, grid_step):
    intervals = []
    interval_step = (end - start)/n_intervals
    for k in range(n_intervals):
        intervals.append(np.linspace(start + interval_step * k, start + interval_step * (k+1),
                                     num=math.ceil(interval_step/grid_step), endpoint=False))
    return intervals


def one_iteration(N_nodes, initial_opinions, parameter_range, index):
    from simulation import Simulation
    from deffuant_polar import DeffuantModelPolar

    def run(parameter, initial_value):
        # Initiate a model with confidence bound specified in parameter
        confidence_bound, cautiousness = parameter, 0.5
        model = DeffuantModelPolar(N_nodes, confidence_bound, cautiousness)
        # Set initial condition
        model.set_circled_opinion(initial_value)
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
        return initial_opinions

    # we split work between processes by parameter
    # index is a process specific part of the job
    def parameter_iterator():
        return parameter_range[index]

    generator = Simulation(parameter_iterator, initial_values_iterator, run)

    return generator.run()


def unpack(lists):
    merged_lists = {**lists[0], **lists[1], **lists[2], **lists[3], **lists[4], **lists[5], **lists[6],
                    **lists[7], **lists[8], **lists[9]}
    return merged_lists


# Create a set of parameter intervals
step = 0.01
parameter_range = get_intervals(0.1, 0.3, 10, step)

total_cores = 4
p = Pool(total_cores)
t0 = time.time()

f = partial(one_iteration, N_nodes, initial_opinions, parameter_range)

try:
    # experiments = unpack(p.map(f, range(len(intervals))))
    experiments = unpack(p.map(f, range(len(parameter_range))))
finally:
    p.terminate()

t1 = time.time()
convergence_time = t1-t0
print("performance time", convergence_time)

# Writing to json
data = {'setup': {'N_nodes': N_nodes,
                  'step': step,
                  'notes': 'polar model with uniform IC',
                  'convergence_time': convergence_time},
        'experiments': experiments,
        'initial_conditions': [r.tolist() for r in initial_opinions]
        }

filename = '/Users/daxelka/Research/Deffuant_model/ABM_simulation/data/polar.txt'

with open(filename, 'w') as outfile:
    json.dump(data, outfile)
print('json created')