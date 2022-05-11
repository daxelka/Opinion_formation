import numpy as np
import json
import math
import matplotlib.pyplot as plt


def unpack_json(filename):
    x = []
    y = []
    p = []
    with open(filename) as json_file:
        data = json.load(json_file)
        for parameter, results in data['experiments'].items():
            for r in results:
                for c in r:
                    if c[1] > 0.1:
                        x.append(float(parameter))
                        y.append(c[0])
                        p.append(c[1])
    return data, x, y, p


def transform_to_delta_grid(x, y, is_polar = False):
    x_t = 0.5 / np.array(x)
    if ~is_polar:
        y_t = (np.array(y) - 0.5) / np.array(x)
    else:
        y_t = (np.array(y)/2/math.pi - 0.5)/np.array(x)
    return x_t, y_t


def unpack_n_clusters(filename):
    x = []
    y = []
    p = []
    with open(filename) as json_file:
        data = json.load(json_file)
        for parameter, results in data['experiments'].items():
            for r in results:
                n_clusters = 0
                for c in r:
                    if c[1] > 0.7 * float(parameter) / 0.5:
                        n_clusters += 1
                x.append(float(parameter))
                y.append(n_clusters)
    return data, x, y, p


def run_and_record(model, n_steps, start_point_recording, time_interval):
    opinions = []
    time_sample = []
    # Run n steps
    k = 0
    for time_step in range(n_steps):
        model.single_step()
        k = k + 1
        if time_step >= start_point_recording and k >= time_interval:
            new_opinion = model.get_unconverged_opinion()
            opinions.append(list(new_opinion))
            time_sample.append(time_step)
            k = 0
    return opinions, time_sample


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
        entropy = -np.sum(np.log(counts_nonzero/ N) * counts_nonzero / N)
        entropy_arr.append(entropy)

    # averaged entropy
    entropy_avg = np.sum(np.array(entropy_arr)) / len(entropy_arr)
    coefficient = np.e ** entropy_avg / n_bins
    return coefficient
