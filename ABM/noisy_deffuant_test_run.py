import numpy as np
import matplotlib.pyplot as plt
import time
from ABM.deffuant_polar import DeffuantModelPolar
import distribution_tools as tools
from utils import libs
import math
import random


# Initiating a opinions
N_nodes: int = int(1e03)
n_steps: int = int(4e06)

t0 = time.perf_counter()

# Plotting parameters
N_sample = int(1e02)
time_interval = int(1e03)
start_point_recording = int(1e06)
end_point_recording = n_steps

# initial_opinion = tools.uniform_opinion(N_nodes, limits=(0.0, 2*math.pi))
initial_opinion_flat = tools.uniform_opinion(N_nodes, limits=(0.0, 1.0))
initial_opinion = 2*math.pi * initial_opinion_flat

# Initiate the model
model = DeffuantModelPolar(N_nodes, confidence_interval=0.28,
                           cautiousness=0.5,
                           jump_radius=0.1,
                           jump_frequency=0.1)

# Set initial conditions in circled space
model.set_opinion(initial_opinion)

# Run one step
opinions = []
opinions.append(list(model.get_unconverged_opinion()))

# Run n steps
for i in range(n_steps):
    model.single_step()

    if i > start_point_recording:
        new_opinion = model.get_unconverged_opinion()
        opinions.append(list(new_opinion))

t1 = time.perf_counter()
print('done, perfomance time: ', round(t1-t0, 2))
# model.show_opinion_distribution(opinions[-1])

# Sample N_sample nodes for illustration

time_sample = list(range(1, end_point_recording - start_point_recording, time_interval))
# time_sample = list(range(time_start_point, time_end_point, time_interval))
opinion_sample = random.sample(range(N_nodes), N_sample)
opinions_np = np.array(opinions)
opinions_selected = []
# Sampling N_sample nodes opinions
[opinions_selected.append(list(opinions_np[i, opinion_sample])) for i in time_sample]
print('opinions selected')

# Creating time list
time_variable = []
[time_variable.append(list(np.ones(N_sample) * time)) for time in time_sample]

plt.scatter(time_variable, opinions_selected, s=0.1, c='black')
plt.ylim(0, 6.28)
# plt.plot(time_variable, opinions_selected, 'o-')
plt.show()
