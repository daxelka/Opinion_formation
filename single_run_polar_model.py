import networkx as nx
import time
from deffuant_polar import DeffuantModelPolar
from distribution_tools import normal_opinion
# from distribution_tools import uniform_opinion
import distribution_tools as tools
from distribution_tools import inverse_transform_sampling
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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

initial_opinion = tools.uniform_opinion(N_nodes)

# Initiate the model
model = DeffuantModelPolar(N_nodes, 0.5, 0.5)

# Set initial conditions in circled space
model.set_circled_opinion(initial_opinion)

model.show_opinion_distribution(model.get_unconverged_opinion())

# # Run one step
opinions = []
opinions.append(list(model.get_unconverged_opinion()))

for i in range(50000):
    new_opinion = model.one_step()
    opinions.append(list(new_opinion))

print('done')
model.show_opinion_distribution(opinions[-1])



fig, ax = plt.subplots(1, 2, subplot_kw=dict(projection='polar'))
tools.circular_hist(ax[0], np.array(opinions[-1]), bins = 100)
plt.show()


# sns.set_style("white")
# # sns.displot(data=np.array(opinions[-1]), kind="kde")
# sns.displot(data=np.array(opinions[-1]), kde=True)
# plt.xlim(0, 6.28)
# plt.show()
