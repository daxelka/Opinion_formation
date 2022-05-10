import numpy as np
import matplotlib.pyplot as plt
import time
from ABM.deffuant_polar import DeffuantModelPolar
import distribution_tools as tools
from utils import libs
import math
import random

# Allowing to use Latex in annotations
plt.rc('text', usetex=True)

# model parameters
N_nodes: int = int(1e03)
epsilon = 0.1
gamma = 0.05
m = 0.01


# simulation parameters
n_steps: int = int(4e06)

# Plotting parameters
N_sample = int(1e02)
time_interval = int(1e03)
start_point_recording = int(5e04)
end_point_recording = n_steps

# Initial opinion
t0 = time.perf_counter()
initial_opinion_flat = tools.uniform_opinion(N_nodes, limits=(0.0, 1.0))
initial_opinion = 2*math.pi * initial_opinion_flat

# Initiate the model
model = DeffuantModelPolar(N_nodes, confidence_interval=epsilon,
                           cautiousness=0.5,
                           jump_radius=gamma,
                           jump_frequency=m)

# Set initial conditions in circled space
model.set_opinion(initial_opinion)

# Run one step
opinions = []
opinions.append(list(model.get_unconverged_opinion()))

time_sample = []
time_sample.append(0)

# Run n steps
k = 0
for time_step in range(n_steps):
    model.single_step()
    k = k + 1
    if time_step >= start_point_recording and k >= time_interval:
        new_opinion = model.get_unconverged_opinion()
        opinions.append(list(new_opinion))
        time_sample.append(time_step)
        k = 0

t1 = time.perf_counter()
print('done, perfomance time: ', round(t1-t0, 2))

# Sample N_sample nodes for illustration
opinion_sample = random.sample(range(N_nodes), N_sample)
opinions_np = np.array(opinions)
opinions_selected = []
opinions_selected = opinions_np[:, opinion_sample]
print('opinions selected')

# Creating time list
time_variable = []
[time_variable.append(list(np.ones(N_sample) * time)) for time in time_sample]

plt.scatter(time_variable, opinions_selected, s=0.1, c='black')
plt.ylim(0, 6.28)
plt.xlim(start_point_recording, end_point_recording)
plt.title(r'$\varepsilon:$' + ' ' + str(epsilon) + ', '
          + r'$\gamma:$' + ' ' + str(gamma) + ', '
          + r'$m:$' + ' ' + str(m) + ', '
          + r'$N_{nodes}:$' + ' ' + str(N_nodes),
          fontsize=20)
plt.xlabel('t (MCS)', fontsize=20)
plt.ylabel('opinion', fontsize=20)
plt.xticks(fontsize=14)

# Saving to file
filename = '/Users/daxelka/Research/Deffuant_model/Simulations/img/noisy_e'\
           + str(epsilon)+'_g'+str(gamma)+'_m'+str(m)+'.png'
plt.savefig(filename)

plt.show()
