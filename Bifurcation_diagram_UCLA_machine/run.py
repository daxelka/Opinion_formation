from deffuant_barabasi_class import DeffuantBarabasiModel
import initial_distributions as ic
# import distribution_tools as tools
import numpy as np
import scipy.stats

# Initiating a opinions
N_nodes: int = 1000
N_experts: int = 100

# Initiate the model
model = DeffuantBarabasiModel(N_nodes, 0.22, 0.5)

# Set initial conditions
initial_opinion = ic.two_gaussians_initial_distribution(N_experts, N_nodes, 0.3, 0.7, 0.1)
model.set_opinion(initial_opinion)
# Show initial distribution
model.show_opinion_distribution(initial_opinion)

# Run N steps on the model
# opinions = []
# opinions.append(list(model.get_unconverged_opinion()))

final_opinion = model.opinion_formation()
model.show_opinion_distribution(final_opinion)
clusters, means = model.clusters_detector(final_opinion)
print('done')

# for i in range(100):
#     new_opinion = model.single_step()

# model.show_opinion_distribution(model.get_unconverged_opinion())

# final_opinion = model.get_unconverged_opinion()

# model.show_opinion_distribution(opinions[-1])

# tools.density_plot(np.array(opinions[-1]), x_limits=(0, 1))