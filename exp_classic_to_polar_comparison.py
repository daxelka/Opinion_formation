# Comparison of early dynamics of classical Deffuant model and the polar model
# Purpose: to illustrate how the extremists shape the dynamics of classical model
# and that this is not observed in the model on the infinite domain

import numpy as np
from deffuant_simple import DeffuantModelSimple
from deffuant_polar import DeffuantModelPolar
import distribution_tools as tools
import math

# Experiment setup
N_nodes: int = 5000
confidence_bound = 0.2
N_steps =  int(5 * N_nodes / confidence_bound)

# Initial opinion distributin
initial_opinion_classic = tools.uniform_opinion(N_nodes)

# Simulation on classical model
classic = DeffuantModelSimple(N_nodes, confidence_bound, 0.5)
classic.set_opinion(list(initial_opinion_classic))

# Run N steps on the model
opinions_classic = []
opinions_classic.append(list(classic.get_unconverged_opinion()))

for i in range(N_steps):
    new_opinion = classic.one_step()
    opinions_classic.append(list(new_opinion))


# Simulation on polar model
initial_opinion_polar = tools.uniform_opinion(N_nodes, limits=(0.0, 2*math.pi))
polar = DeffuantModelPolar(N_nodes, confidence_bound, 0.5)
polar.set_opinion(list(initial_opinion_polar))

# Run N steps on the model
opinions_polar = []
opinions_polar.append(list(polar.get_unconverged_opinion()))

for i in range(N_steps):
    new_opinion = polar.one_step()
    opinions_polar.append(list(new_opinion))

# Comparing results
tools.density_plot(np.array(opinions_classic[-1]), x_limits=(0, 1))
tools.density_plot(np.array(opinions_polar[-1])/2/math.pi, x_limits=(0, 1))

# tools.density_plot(np.array(opinions_classic[10000]), x_limits=(0, 1), title='t = 1.2', x_label='confidence bound')
