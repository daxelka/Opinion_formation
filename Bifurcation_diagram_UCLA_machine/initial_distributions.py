import numpy as np
import scipy


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


def two_gaussians(N, mu1, mu2, sigma):
    # boundaries of the domain
    lower, upper = 0, 1
    # Generate random samples from the two Gaussian distributions
    samples1 = scipy.stats.truncnorm.rvs(
        (lower - mu1) / sigma, (upper - mu1) / sigma, loc=mu1, scale=sigma, size=int(N/2))
    samples2 = scipy.stats.truncnorm.rvs(
        (lower - mu2) / sigma, (upper - mu2) / sigma, loc=mu2, scale=sigma, size=int(N/2))

    # Combine the samples from both distributions
    samples = np.concatenate((samples1, samples2))
    # Shuffle the samples to ensure randomness
    np.random.shuffle(samples)
    return samples


def two_gaussians_initial_distribution(N_experts, N_total, mu1, mu2, sigma):
    array = np.empty(N_total)
    array[:N_experts] = two_gaussians(N_experts, mu1, mu2, sigma)
    array[N_experts:] = np.nan
    np.random.shuffle(array)
    return array
