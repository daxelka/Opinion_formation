from deffuant_barabasi_class import DeffuantBarabasiModel
import distribution_tools as tools
import numpy as np

# Initiating a opinions
N_nodes: int = 10

# initial_opinion = tools.uniform_opinion(N_nodes)
initial_opinion = [np.nan, np.nan, np.nan, np.nan, np.nan, 0.2, 0.3, 0.5, 0.7, 0.8]
# Initiate the model
model = DeffuantBarabasiModel(N_nodes, 0.5, 0.5)

# Set initial conditions
model.set_opinion(initial_opinion)
# model.show_opinion_distribution(initial_opinion)

# Run N steps on the model
# opinions = []
# opinions.append(list(model.get_unconverged_opinion()))

for i in range(10):
    new_opinion = model.single_step()

# model.show_opinion_distribution(opinions[-1])

# tools.density_plot(np.array(opinions[-1]), x_limits=(0, 1))