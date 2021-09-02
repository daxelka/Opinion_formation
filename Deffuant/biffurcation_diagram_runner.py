import networkx as nx
from utils.bifurcation_diagram.generator import BifurcationDiagramGenerator
from utils.bifurcation_diagram.plotter import BifurcationDiagramPlotter
from utils.libs import drange
from deffuant import DeffuantModel
from distribution_tools import uniform_opinion

def parameter_iterator():
    return drange(0.2, 0.6, 0.005)

def initial_values_iterator():
    initial_opinions = []
    for i in range(50):
        distribution = uniform_opinion(N_nodes)
        initial_opinions.append(distribution)
    return initial_opinions


def run(parameter, initial_value):
    model = DeffuantModel(G, parameter, 0.5)
    model.set_opinion(initial_value)
    model.opinion_formation()
    _, means = model.clusters_detector(model.get_opinion())
    return means

# Graph Inisialisation
N_nodes: int = 100
G = nx.complete_graph(N_nodes)

generator = BifurcationDiagramGenerator(parameter_iterator, initial_values_iterator, run)

x_var, y_var = generator.run()
BifurcationDiagramPlotter().plot(x_var, y_var)
