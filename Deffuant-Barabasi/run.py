from deffuant_barabasi_class import DeffuantBarabasiModel
import distribution_tools as tools
import numpy as np

# Initiating a opinions
N_nodes: int = 500

initial_opinion = tools.uniform_opinion(N_nodes)

# Initiate the model
model = DeffuantBarabasiModel(N_nodes, 0.2, 0.5)

# Set initial conditions
model.set_opinion(initial_opinion)
model.show_opinion_distribution(initial_opinion)

# Run N steps on the model
opinions = []
opinions.append(list(model.get_unconverged_opinion()))

for i in range(5):
    new_opinion = model.insert_new_node()

# model.show_opinion_distribution(opinions[-1])

# tools.density_plot(np.array(opinions[-1]), x_limits=(0, 1))