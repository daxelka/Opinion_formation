import networkx as nx
import numpy as np
import math
import json
import time
from simulation import Simulation
from deffuant_simple import DeffuantModelSimple
from distribution_tools import uniform_opinion
from distribution_tools import inverse_transform_sampling


lower_bound = 1
upper_bound = 7
step = 0.05
size = math.ceil((upper_bound - lower_bound)/step)
print(size)

def parameter_iterator():
    deltas = np.linspace(lower_bound, upper_bound, num=size, endpoint=True)
    return 0.5 / deltas

def gen_pdf(n_peaks, epsilon):
    def pdf(x):
        f = np.ones(x.shape) + epsilon * np.cos((2*n_peaks)*np.pi*x)
        f / np.trapz(f, x)
        return f
    return pdf


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


t0 = time.time()

# Graph Initialisation
N_nodes: int = 100

# Generating the set of initial distributions
initial_opinions = []
# Number of run for each parameter
n_runs = 30

for k in range(n_runs):
    distribution = uniform_opinion(N_nodes)
    initial_opinions.append(distribution)


def initial_values_iterator():
    return initial_opinions


# Simulation
generator = Simulation(parameter_iterator, initial_values_iterator, run)

experiments = generator.run()

t1 = time.time()
print("performance time", t1-t0)

# Writing to file
data = {'setup': {'N_nodes': N_nodes,
                  'parameter_limits': (lower_bound, upper_bound),
                  'step': step,
                  'notes': 'uniform IC plus cos disturbance with 2 negative peaks and amplitude 0.5 '},
        'experiments': experiments,
        'initial_conditions': [r.tolist() for r in initial_opinions]
        }

filename = '/Users/daxelka/Research/Deffuant_model/ABM_simulation/data/test.txt'

with open(filename, 'w') as outfile:
    json.dump(data, outfile)
print('json created')
