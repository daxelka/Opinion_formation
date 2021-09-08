import numpy as np
import scipy.stats as stats
import scipy.optimize


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


def normal_opinion(n_nodes, mu, sigma, lower_bound, upper_bound):
    mu_to_use, sigma_to_use = mu, sigma
    # mu_to_use, sigma_to_use = corrector(mu, sigma, lower_bound, upper_bound)
    opinion = trunc_samples(mu=mu_to_use, sigma=sigma_to_use, lower=lower_bound, upper=upper_bound, num_samples=n_nodes)
    return opinion


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
