import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

def draw_nodes(N, mu1, mu2, sigma):
    # boundaries of the domain
    lower = 0
    upper = 1

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


samples = draw_nodes(1000, 0.3, 0.7, 0.1)
plt.hist(samples)
plt.show()

