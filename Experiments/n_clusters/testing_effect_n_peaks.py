import numpy as np
import time
from utils.bifurcation_diagram.plotter import BifurcationDiagramPlotter
from utils.libs import drange
from ABM.deffuant_simple import DeffuantModelSimple
from distribution_tools import inverse_transform_sampling


class DiagramGenerator:
    def __init__(self, parameter_iterator, initial_values_iterator, model_run):
        self.parameter_iterator = parameter_iterator
        self.initial_values_iterator = initial_values_iterator
        self.model_run = model_run

    def run(self):
        results = []
        for parameter in self.parameter_iterator():
            for initial_value in self.initial_values_iterator(parameter):
                converged = self.model_run(initial_value)  # can be list or np.array ?
                r = list(map(lambda c: [parameter, c], converged))
                results = results + r

        x_var = [x[0] for x in results]
        y_var = [x[1] for x in results]
        return x_var, y_var


def parameter_iterator():
    return range(1, 10)


def gen_pdf(n_peaks, epsilon):
    def pdf(x):
        f = np.ones(x.shape) + epsilon * np.cos((2*n_peaks-1)*np.pi*x)
        f / np.trapz(f, x)
        return f
    return pdf


def initial_values_iterator(parameter):
    initial_opinions = []
    for amplitude in drange(0.1, 0.6, 0.05):
        for i in range(5):
            pdf = gen_pdf(parameter, amplitude)
            distribution = inverse_transform_sampling(pdf, N_nodes, (0, 1))
            initial_opinions.append(distribution)

    return initial_opinions


def run(initial_value):
    confidence_bound = 0.3
    model = DeffuantModelSimple(initial_value, confidence_bound, 0.5)
    model.opinion_formation()
    clusters, means = model.clusters_detector(model.get_opinion())
    # Filter by cluster density
    densities = model.cluster_density(clusters)
    major_groups = np.array(means)[np.array(densities) > 0.1]
    return major_groups


N_nodes: int = 1000

t0 = time.time()
generator = DiagramGenerator(parameter_iterator, initial_values_iterator, run)

x_var, y_var = generator.run()
t1 = time.time()
print("performance time", t1-t0)

BifurcationDiagramPlotter().plot(x_var, y_var, '# peaks', 'opinion clusters')

