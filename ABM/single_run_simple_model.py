from ABM.deffuant_simple import DeffuantModelSimple
import distribution_tools as tools
import numpy as np

# Initiating a opinions
N_nodes: int = 500

# def gen_pdf(n_peaks, epsilon):
#     def pdf(x):
#         f = np.ones(x.shape) + epsilon * np.cos((2*n_peaks-1)*np.pi*x)
#         f / np.trapz(f, x)
#         return f
#     return pdf



# pdf = gen_pdf(3, 0.1)

initial_opinion = tools.uniform_opinion(N_nodes)

# Initiate the model
model = DeffuantModelSimple(N_nodes, 0.2, 0.5)

# Set initial conditions
model.set_opinion(initial_opinion)
model.show_opinion_distribution(initial_opinion)

# Run N steps on the model
opinions = []
opinions.append(list(model.get_unconverged_opinion()))

for i in range(5):
    new_opinion = model.one_step()
    opinions.append(list(new_opinion))

# clusters, means = model.clusters_detector(model.get_opinion())
# densities = model.cluster_density(clusters)
# print('menas:', means)
# print('densities:',densities)
#
#
# # Clusters in final opinion
# clusters, means = model.clusters_detector(model.get_opinion())
# print('Means of clusters:', means)

#Show opinion distribution
# model.show_opinion_distribution(model.get_opinion())
model.show_opinion_distribution(opinions[-1])

tools.density_plot(np.array(opinions[-1]), x_limits=(0, 1))