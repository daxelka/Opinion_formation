import numpy as np
import matplotlib.pyplot as plt
import time
from ABM.deffuant_polar import DeffuantModelPolar
from ABM.deffuant_simple import DeffuantModelSimple
import distribution_tools as tools
from data_tools import cluster_coefficient
from data_tools import run_and_record
from utils import libs
import math
import random

# Allowing to use Latex in annotations
plt.rc('text', usetex=True)

# model parameters
N_nodes: int = int(1e04)
epsilon = 0.5
gamma = 0.1
m = 0.01

# simulation parameters
n_steps: int = int(100)

# Plotting parameters
N_sample = N_nodes
time_interval = int(1)
start_point_recording = int(1)
end_point_recording = n_steps

# Initial opinion
t0 = time.perf_counter()
initial_opinion_flat = tools.uniform_opinion(N_nodes, limits=(0.0, 1.0))
initial_opinion = 2 * math.pi * initial_opinion_flat
# initial_opinion = initial_opinion_flat

# Initiate the model
model = DeffuantModelPolar(N_nodes, confidence_interval=epsilon,
                           cautiousness=0.5,
                           jump_radius=gamma,
                           jump_frequency=m)
# model = DeffuantModelSimple(N_nodes, confidence_interval=epsilon,
#                            cautiousness=0.5,
#                            jump_radius=gamma,
#                            jump_frequency=m)

# Set initial conditions in circled space
model.set_opinion(initial_opinion)


# Run model and record
opinions, time_sample = run_and_record(model, n_steps, start_point_recording, time_interval)

t1 = time.perf_counter()
print('done, perfomance time: ', round(t1 - t0, 2))

# Creating time list
time_variable = []
[time_variable.append(list(np.ones(N_sample) * time)) for time in time_sample]

plt.scatter(time_variable, opinions, s=0.1, c='black')
plt.ylim(0, 6.28)
plt.xlim(start_point_recording, end_point_recording)
plt.show()


# Calculate cluster coefficient
opinions_np = np.array(opinions)
time_dim_size = np.shape(opinions_np)[0]
opinion_dim_size = np.shape(opinions_np)[1]
Gm = cluster_coefficient(opinions_np, n_bins=100, time_dim_size=time_dim_size, opinion_dim_size=opinion_dim_size,
                         opinion_min=0, opinion_max=6.28)
print(Gm)
