from deffuant_barabasi_class import DeffuantBarabasiModel
import distribution_tools as tools
import numpy as np

# Initiating a opinions
N_nodes: int = 500
N_experts: int = 50

# Initiate the model
model = DeffuantBarabasiModel(N_nodes, 0.25, 0.5)

# Set initial conditions

def create_initial_distribution(N_experts, N_total):
    array = np.empty(N_total)
    array[:N_experts] = np.random.uniform(size=N_experts)
    array[N_experts:] = np.nan
    np.random.shuffle(array)
    return array

def evenly_spaced_initial_conditions(N_experts, N_total):
    array = np.empty(N_total)
    # Generate N equally spaced values between 0 and 1
    array[:N_experts] = np.linspace(0, 1, N_experts)
    array[N_experts:] = np.nan
    np.random.shuffle(array)
    return array



# initial_opinion = tools.uniform_opinion(N_nodes)
# initial_opinion = [np.nan, np.nan, np.nan, np.nan, np.nan, 0.2, 0.3, 0.5, 0.7, 0.8]
# initial_opinion = create_initial_distribution(N_experts, N_nodes)
initial_opinion = evenly_spaced_initial_conditions(N_experts, N_nodes)
model.set_opinion(initial_opinion)
model.show_opinion_distribution(initial_opinion)

# Run N steps on the model
# opinions = []
# opinions.append(list(model.get_unconverged_opinion()))

final_opinion = model.opinion_formation()
model.show_opinion_distribution(final_opinion)
# model.clusters_detector(final_opinion)
clusters, means = model.clusters_detector(final_opinion)

# for i in range(100):
#     new_opinion = model.single_step()

# model.show_opinion_distribution(model.get_unconverged_opinion())

# final_opinion = model.get_unconverged_opinion()

# model.show_opinion_distribution(opinions[-1])

# tools.density_plot(np.array(opinions[-1]), x_limits=(0, 1))