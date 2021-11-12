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
n_runs = 20

# Create a set of initial distributions
initial_opinions = []
n_peaks = 3
amplitude = 0.1

for sigma in range(n_runs):
    pdf = gen_pdf(n_peaks, amplitude)
    distribution = inverse_transform_sampling(pdf, N_nodes, (0, 1))
    initial_opinions.append(distribution)

# show_distribution(initial_opinions[10])
# print('done')

# Create a set of parameter intervals
intervals = [(1, 1.5), (1.5, 1.75), (1.75, 2), (2, 2.25), (2.25, 2.5), (2.5, 2.75), (2.75, 3),
             (3, 3.25), (3.25, 3.5), (3.5, 3.75), (3.75, 4), (4, 4.25), (4.25, 4.5), (4.5, 4.75), (4.75, 5)]
step = 0.05

parameter_range = []
for interval in intervals:
    start, end = interval
    size = math.ceil((end - start) / step)
    deltas = np.linspace(start, end, num=size, endpoint=False)
    epsilons = 0.5 / deltas
    parameter_range.append(epsilons)


def one_iteration(N_nodes, initial_opinions, parameter_range, index):
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
        return initial_opinions

    # we split work between processes by parameter
    # index is a process specific part of the job
    def parameter_iterator():
        # start = 0.1 + (0.5 - 0.1) / total_cores * index
        # end = 0.1 + (0.5 - 0.1) / total_cores * (index + 1)
        # return drange(start, end, 0.01)
        return parameter_range[index]

    generator = Simulation(parameter_iterator, initial_values_iterator, run)

    return generator.run()


def unpack(lists):
    merged_lists = {**lists[0], **lists[1], **lists[2], **lists[3], **lists[4], **lists[5], **lists[6],
                    **lists[7], **lists[8], **lists[9], **lists[10], **lists[11], **lists[12], **lists[13], **lists[14]}
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
                  'notes': 'uniform IC plus cos disturbance with ' + str(n_peaks) + ' negative peaks and amplitude ' + str(amplitude),
                  'convergence_time': convergence_time},
        'experiments': experiments,
        'initial_conditions': [r.tolist() for r in initial_opinions]
        }

filename = '/Users/daxelka/Research/Deffuant_model/ABM_simulation/data/cos_'+str(n_peaks)+'peaks_'+str(amplitude)+'ampl.txt'

with open(filename, 'w') as outfile:
    json.dump(data, outfile)
print('json created')