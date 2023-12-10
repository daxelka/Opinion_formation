import math
from pathos.multiprocessing import ProcessingPool as Pool
import time
import json

import distribution_tools as tools
import numpy as np
from functools import partial

# Graph Initialisation
N_nodes: int = 1000
N_experts: int = 100
# Number of runs for each parameter
n_runs = 5

# Create a set of initial distributions
initial_opinions = []

def create_initial_distribution(N_experts, N_total):
    array = np.empty(N_total)
    array[:N_experts] = np.random.uniform(size=N_experts)
    array[N_experts:] = np.nan
    np.random.shuffle(array)
    return array

def evenly_spaced_initial_conditions(N_experts, N_total):
    array = np.empty(N_total)
    # Generate N equally spaced values between 0 and 1
    # array[:N_experts] = np.linspace(0, 1, N_experts)
    array[:N_experts] = np.repeat(np.linspace(0, 1, int(N_experts/5)), 5)
    array[N_experts:] = np.nan
    np.random.shuffle(array)
    return array

for sigma in range(n_runs):
    distribution = create_initial_distribution(N_experts, N_nodes)
    initial_opinions.append(distribution)

# show_distribution(initial_opinions[10])
# print('done')

# Create a set of parameter intervals
intervals = [(1, 1.5), (1.5, 1.75), (1.75, 2), (2, 2.25), (2.25, 2.5), (2.5, 2.75), (2.75, 3),
             (3, 3.25), (3.25, 3.5), (3.5, 3.75), (3.75, 4)]
step = 0.1

parameter_range = []
for interval in intervals:
    start, end = interval
    size = math.ceil((end - start) / step)
    deltas = np.linspace(start, end, num=size, endpoint=False)
    epsilons = 0.5 / deltas
    parameter_range.append(epsilons)


def one_iteration(N_nodes, initial_opinions, parameter_range, index):
    from ABM.simulation import Simulation
    from deffuant_barabasi_class import DeffuantBarabasiModel

    def run(parameter, initial_value):
        # Initiate a model with confidence bound specified in parameter
        confidence_bound, cautiousness = parameter, 0.5
        model = DeffuantBarabasiModel(N_nodes, confidence_bound, cautiousness)
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
        return initial_opinions

    # we split work between processes by parameter
    # index is a process specific part of the job
    def parameter_iterator():
        return parameter_range[index]

    generator = Simulation(parameter_iterator, initial_values_iterator, run)

    return generator.run()


def unpack(lists):
    merged_lists = {**lists[0], **lists[1], **lists[2], **lists[3], **lists[4], **lists[5], **lists[6],
                    **lists[7], **lists[8], **lists[9], **lists[10]}
    return merged_lists


total_cores = 6
p = Pool(total_cores)
t0 = time.time()

f = partial(one_iteration, N_nodes, initial_opinions, parameter_range)

try:
    experiments = unpack(p.map(f, range(len(intervals))))
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

filename = '/Users/daxelka/Research/Deffuant_model/Deffuant-Barabasi/biff_DB_1000_evenly_spaced_repeated_5.txt'

with open(filename, 'w') as outfile:
    json.dump(data, outfile)
print('json created')