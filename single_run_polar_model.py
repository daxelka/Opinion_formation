import numpy as np
import matplotlib.pyplot as plt
import time
from deffuant_polar import DeffuantModelPolar
import distribution_tools as tools
import math



# Initiating a opinions
N_nodes: int = 1000

# def gen_pdf(n_peaks, epsilon):
#     def pdf(x):
#         f = np.ones(x.shape) + epsilon * np.cos((2*n_peaks-1)*np.pi*x)
#         f / np.trapz(f, x)
#         return f
#     return pdf


t0 = time.perf_counter()
# pdf = gen_pdf(3, 0.1)

# initial_opinion = tools.uniform_opinion(N_nodes, limits=(0.0, 2*math.pi))
initial_opinion_flat = tools.uniform_opinion(N_nodes, limits=(0.0, 1.0))
initial_opinion = 2*math.pi * initial_opinion_flat

# Initiate the model
model = DeffuantModelPolar(N_nodes, 0.2, 0.5)

# Set initial conditions in circled space
model.set_opinion(initial_opinion)

model.show_opinion_distribution(model.get_unconverged_opinion())
#
# Run one step
opinions = []
opinions.append(list(model.get_unconverged_opinion()))

for i in range(50000):
    new_opinion = model.one_step()
    opinions.append(list(new_opinion))

print('done')
model.show_opinion_distribution(opinions[-1])

tools.density_plot(np.array(opinions[-1])/2/math.pi, x_limits=(0, 1))

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(4, 4))
# Plot origin (agent's start point)
# ax.plot(0, 0, color='black', marker='o', markersize=5)
tools.circular_hist(ax, np.array(opinions[-1]), bins = 500)
plt.show()

# fig, ax = plt.subplots(1, 2, subplot_kw=dict(projection='polar'))
# tools.circular_hist(ax[0], np.array(opinions[-1]), bins = 100)
# plt.show()