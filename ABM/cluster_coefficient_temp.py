import matplotlib.pyplot as plt
import numpy as np

N = 100
x = []
for i in range(10):
    x.append(list(np.random.uniform(0, 1, N)))


def cluster_coefficient(opinions, n_bins, time_dim_size, opinion_dim_size, opinion_min=0, opinion_max=1):
    """

    Args:
        opinions: np.array
        n_bins:
        time_dim_size:
        opinion_dim_size:
        opinion_min: left border of opinion space
        opinion_max: right border of opinion space

    Returns:
        cluster coefficient with averaged entropy (float in [1/n_bins, 1])
    """

    N = opinion_dim_size
    entropy_arr = []
    bins_edges = np.linspace(opinion_min, opinion_max, n_bins, endpoint=True)

    for t in range(time_dim_size):
        counts, __, __ = plt.hist(opinions[t, :], bins=bins_edges)
        counts_nonzero = counts[counts > 0]
        entropy = np.sum(np.log(counts_nonzero/ N) * counts_nonzero / N)
        entropy_arr.append(entropy)

    # averaged entropy
    entropy_avg = np.sum(np.array(entropy_arr)) / len(entropy_arr)
    coefficient = np.e ** entropy_avg / n_bins
    return coefficient


x_np = np.array(x)
time_dim_size = np.shape(x_np)[0]
opinion_dim_size = np.shape(x_np)[1]
Gm = cluster_coefficient(x_np, n_bins=5, time_dim_size=time_dim_size, opinion_dim_size=opinion_dim_size)
print(Gm)