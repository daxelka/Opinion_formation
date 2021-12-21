import matplotlib.pyplot as plt
import numpy as np
import math
from data_tools import transform_to_delta_grid

epsilons = np.linspace(0.1, 0.2, 100, endpoint=True)

k_const = 2.85
x = []
y = []
# n_max_array = []


def perturbation(k,e):
    # return 4 * math.sin(k_s/2)/k_s - math.sin(k_s)/k_s - 1
    return 8 * math.sin(k*e/2)/k - 2 * math.sin(k*e)/k - 2*e


def fastest_mode(e):
    n_max = math.floor(k_const / math.pi * (1 / 2 / e - 1))
    k_n = 2*n_max*math.pi/(1-2*e)
    k_nn = 2*(n_max+1)*math.pi/(1-2*e)

    if perturbation(k_n, e) >= perturbation(k_nn, e):
        print(1)
        return n_max
    else:
        print(2)
        return n_max + 1
    # return int(1/2/e)-1


def cluster_position(e, n, n_max):
    return e + n*(1-2*e)/n_max


for e in epsilons:
    n_max = fastest_mode(e)
    # n_max_array.append(n_max+1)
    print('e:', e, ' n_max:', n_max)
    for i in range(n_max+1):
        cluster = cluster_position(e, i, n_max)
        x.append(e)
        y.append(cluster)


x_d, y_d = transform_to_delta_grid(x, y)
plt.scatter(x_d, y_d)
plt.show()
