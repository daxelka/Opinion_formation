import networkx as nx
import numpy as np
import math
import json
import time
import pickle
from utils.bifurcation_diagram.generator import BifurcationDiagramGenerator
from utils.bifurcation_diagram.plotter import BifurcationDiagramPlotter
from utils.libs import drange
from deffuant_simple import DeffuantModelSimple
from distribution_tools import normal_opinion
from distribution_tools import uniform_opinion
from distribution_tools import inverse_transform_sampling


lower_bound = 1
upper_bound = 5
step = 0.01
# size = math.ceil((upper_bound - lower_bound)/step)
size = 10

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
    confidence_bound, cautiousness = parameter, 0.5
    model = DeffuantModelSimple(N_nodes, confidence_bound, cautiousness)
    model.set_opinion(initial_value)
    model.opinion_formation()
    # if model converged to return opinion if not to return []
    clusters, means = model.clusters_detector(model.get_opinion())

    # Filter by cluster density
    densities = model.cluster_density(clusters)
    major_groups = np.array(means)[np.array(densities) > 0.1]

    # Writing result into data
    results = []
    for count, mean_value in enumerate(means):
        density = densities[count]
        results.append([mean_value, density])

    data['experiments'].append({
        'parameter': parameter,
        'results': results
    })
    return major_groups


t0 = time.time()

# Graph Initialisation
N_nodes: int = 100

# Generating the set of initial distributions
initial_opinions = []
n_peaks = 2

for k in range(5):
    pdf = gen_pdf(n_peaks, 0.5)
    distribution = inverse_transform_sampling(pdf, N_nodes, (0, 1))
    initial_opinions.append(distribution)


def initial_values_iterator():
    return initial_opinions


data = {'setup': {'N_nodes': N_nodes,
                  'parameter_limits': (lower_bound, upper_bound),
                  'step': step,
                  'notes': 'uniform IC plus cos disturbance with 2 negative peaks and amplitude 0.5 '},
        'experiments': [],
        'initial_conditions': [r.tolist() for r in initial_opinions]
        }

generator = BifurcationDiagramGenerator(parameter_iterator, initial_values_iterator, run)

x_var, y_var = generator.run()
t1 = time.time()
print("performance time", t1-t0)

# transformation to (y, delta)
y = (np.array(y_var) - 0.5)/np.array(x_var)
x = 0.5/np.array(x_var)

# BifurcationDiagramPlotter().plot(x_var, y_var, 'confidence bound', 'opinion')
BifurcationDiagramPlotter().plot(x, y, 'confidence bound', 'opinion', y_limits = (-5,5))

# Writing to file
with open('/Users/daxelka/Research/Deffuant_model/ABM_simulation/data/data.txt', 'w') as outfile:
    json.dump(data, outfile)

print('json created')