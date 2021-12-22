import numpy as np
import math
import matplotlib.pyplot as plt
from ABM.deffuant_polar import DeffuantModelPolar
import distribution_tools as tools


# Graph Initialisation
N_nodes: int = 100

# Number of run for each parameter
n_runs = 1
n_repetitions = 10
# tools.show_distribution(initial_opinions[0])

# Initiate the model
model = DeffuantModelPolar(N_nodes, 0.49, 0.5)

initial_opinion_flat = tools.uniform_opinion(N_nodes, limits=(0.0, 1.0))
distribution = 2*math.pi * initial_opinion_flat

opinions = []

for m in range(n_repetitions):
    model.set_opinion(list(distribution))
    model.opinion_formation()

    if model.get_opinion():
        opinions.append(list(model.get_opinion()))

print('done')


# # Writing to json
# data = {'setup': {'N_nodes': N_nodes,
#                   'notes': 'polar model with uniform IC'},
#         'experiments': opinions,
#         'initial_conditions': [r.tolist() for r in initial_opinions]
#         }
#
# filename = '/Users/daxelka/Research/Deffuant_model/ABM_simulation/data/experiment3011.txt'
#
# with open(filename, 'w') as outfile:
#     json.dump(data, outfile)
# print('json created')

colors = ['black', 'dimgray', 'gray', 'darkgray', 'silver', 'lightgray',
          'black', 'dimgray', 'gray', 'darkgray', 'silver', 'lightgray',
          'black', 'dimgray', 'gray', 'darkgray',
          'black', 'dimgray', 'gray', 'darkgray', 'silver', 'lightgray',
          'black', 'dimgray', 'gray', 'darkgray', 'silver', 'lightgray',
          'black', 'dimgray', 'gray', 'darkgray',
          ]

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(4, 4))
for count, opinion in enumerate(opinions):
    tools.circular_hist(ax, np.array(opinion), bins= 500)
plt.show()