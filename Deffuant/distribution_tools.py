import numpy as np


def random_opinion(n_nodes):
    rng = np.random.default_rng()
    opinion_distribution = rng.random((n_nodes,))
    return opinion_distribution


def uniform_opinion(n_nodes):
    rng = np.random.default_rng()
    opinion = rng.uniform(0.0, 1.0, (n_nodes,))
    return opinion


# def sin_opinion(n_nodes):
#     # randomly chosen N_nodes numbers from [0,1) from uniform distribution
#     rng = np.random.default_rng()
#     values = rng.uniform(0.0, 1.0, (n_nodes,))
#     m = 1
#     # opinion_distribution = values + m * np.sin(2*math.pi/k * np.linspace(0, 1, N_nodes, endpoint=False))
#     opinion = m * np.arccos(np.linspace(-1, 1, n_nodes)) / math.pi
#     return opinion


def normal_opinion(n_nodes, mu, sigma):
    rng = np.random.default_rng()
    opinion = rng.normal(mu, sigma, n_nodes)
    return opinion


def multimodal_normal_opinion(n_nodes, sigma):
    rng = np.random.default_rng()
    mus = [.25, .75]
    opinion = np.zeros(n_nodes)
    for mu in mus:
        opinion += rng.normal(mu, sigma, n_nodes)
    return opinion