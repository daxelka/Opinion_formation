from ABM.deffuant_barabasi_class import DeffuantBarabasiModel
import distribution_tools as tools

# Initiating a opinions
N_nodes: int = 5

# Initiate the model
model = DeffuantBarabasiModel(N_nodes, 0.2, 0.5)

# Set initial conditions
initial_opinion = tools.uniform_opinion(N_nodes)
print(initial_opinion)
model.set_opinion(initial_opinion)

for i in range(3):
    model.single_step()

print(model.get_unconverged_opinion())
