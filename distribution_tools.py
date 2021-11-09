import numpy as np
import matplotlib.pyplot as plt
from numpy.random import random
import scipy.stats as stats
import scipy.optimize
from scipy import interpolate



def random_opinion(n_nodes):
    rng = np.random.default_rng()
    opinion_distribution = rng.random((n_nodes,))
    return opinion_distribution


def uniform_opinion(n_nodes):
    rng = np.random.default_rng()
    opinion = rng.uniform(0.0, 1.0, (n_nodes,))
    return opinion


def inverse_transform_sampling(pdf, N=1000000, x_limits=(0, 1)):
    """ Sampling of a random variable from a user specified probability density function by inverse transform sampling.

    Args:
        pdf:
            probability density function. Must take a single argument, x, in form of np.array, and return np.array.
        N:
            number of samples. integer
        x_limits:
            boundaries of truncated pdf. tuple (x_lower, x_upper)

    Returns: N random variables sampled from given truncated pdf. np.array of size N

    """
    x_lower, x_upper = x_limits
    x = np.linspace(x_lower, x_upper, int(N))
    y = pdf(x)  # probability density function, pdf
    cdf_y = np.cumsum(y)  # cumulative distribution function
    cdf_y = cdf_y / cdf_y.max()  # normalization of cdf
    inverse_cdf = interpolate.interp1d(cdf_y, x, fill_value="extrapolate")  # this is a function

    # Generates samples from given pdf
    uniform_samples = random(int(N))
    required_samples = inverse_cdf(uniform_samples)
    return required_samples



def normal_opinion(n_nodes, mu, sigma, lower_bound, upper_bound):
    mu_to_use, sigma_to_use = mu, sigma
    # mu_to_use, sigma_to_use = corrector(mu, sigma, lower_bound, upper_bound)
    opinion = trunc_samples(mu=mu_to_use, sigma=sigma_to_use, lower=lower_bound, upper=upper_bound, num_samples=n_nodes)
    return opinion

def show_distribution(opinions):
    bins = [0.01 * n for n in range(100)]
    plt.hist(opinions, bins=bins, density=True)
    plt.title("Histogram of opinions")
    plt.show()

# def multimodal_normal_opinion(n_nodes, sigma):
#     rng = np.random.default_rng()
#     mus = [.25, .75]
#     opinion = np.zeros(n_nodes)
#     for mu in mus:
#         opinion += rng.normal(mu, sigma, n_nodes)
#     return opinion

def truncated_mean_std(mu, sigma, lower, upper):
    # N.B. lower/upper are the actual values, not Z-scaled
    alpha = (lower - mu)/sigma
    beta = (upper - mu)/sigma
    d_pdf = (stats.norm.pdf(alpha) - stats.norm.pdf(beta))
    wd_pdf = (alpha * stats.norm.pdf(alpha) - beta * stats.norm.pdf(beta))
    d_cdf = stats.norm.cdf(beta) - stats.norm.cdf(alpha)
    mu_trunc = mu + sigma * (d_pdf / d_cdf)
    var_trunc = sigma**2 * (1 + wd_pdf / d_cdf - (d_pdf/d_cdf)**2)
    std_trunc = var_trunc**0.5
    return mu_trunc, std_trunc

def trunc_samples(mu, sigma, lower, upper, num_samples=1000):
    n = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    samples = n.rvs(num_samples)
    return samples

def corrector(mu, sigma, lower, upper):
    target = np.array([mu, sigma])
    result = scipy.optimize.minimize(
        lambda x: ((target - truncated_mean_std(x[0], x[1], lower, upper))**2).sum(),
        x0=[mu, sigma])
    return result.x

# mu, sigma = 0, 5 # mean and standard deviation
# s = trunc_samples(mu=mu, sigma=sigma, lower=0, upper=1, num_samples=10**7)
# mu_to_use, sigma_to_use = corrector(mu, sigma, 0, 1)
# print(s.mean(), s.std())
# print(mu_to_use, sigma_to_use)
# s = trunc_samples(mu=mu_to_use, sigma=sigma_to_use, lower=0, upper=1, num_samples=10**7)
# m = normal_opinion(10**7, mu, sigma, 0, 1)
#
# print(s.mean(), s.std())
# print(m.mean(), m.std())
