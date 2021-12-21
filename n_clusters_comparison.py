import matplotlib.pyplot as plt
import numpy as np
import math
from data_tools import transform_to_delta_grid
from data_tools import unpack_n_clusters

deltas = np.linspace(4, 9.5, 100, endpoint=True)
# epsilons = np.linspace(0.07, 0.4, 100, endpoint=True)
epsilons = 1/2/deltas

k_const = 2.85
x = []
y = []
n_max_approx = []
n_max_exact = []
n_12d = []


def perturbation(k, e):
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

def fastest_mode_approx(e):
    n_max = round(k_const / math.pi * (1 / 2 / e - 1)+1)
    return n_max


for e in epsilons:
    # Exact solution
    n_max_exact.append(fastest_mode(e)+1)

    # Approximate Susan's result
    n_max_approx.append(fastest_mode_approx(e))

    # 1/2d rule
    d, dd = math.modf(1/2/e)
    n_12d.append(dd)

# number of clusters from ABM simulations
# filename = '/Users/daxelka/Research/Deffuant_model/ABM_simulation/data/polar_bifurcation_10k_20runs.txt'
filename = '/Users/daxelka/Research/Deffuant_model/ABM_simulation/data/uniform_10k_20runs.txt'

# unpack json
data, e_abm, n_abm, p = unpack_n_clusters(filename)

# deltas_biffurcations = np.array([1.871, 2.248, 4.079, 4.455, 6.259, 6.638, 8.431,8.810])
deltas_biffurcations = np.array([4.079, 4.455, 6.259, 6.638, 8.431, 8.810])
number_clusters = np.array([4,5,6,7,8,9])

plt.scatter(1/2/epsilons, n_max_exact, color='hotpink')
# plt.scatter(1/2/epsilons, n_max_approx, color='#88c999', alpha=0.3)
plt.scatter(1/2/epsilons, n_12d, color='blue', alpha=0.3)
# plt.scatter(1/2/np.array(e_abm), n_abm, color='#88c999', alpha=0.3)

plt.scatter(deltas_biffurcations, number_clusters, color='k')
for i, xc in enumerate(deltas_biffurcations):
    ymin = number_clusters[i] - 0.5
    ymax = number_clusters[i] + 0.5
    print(xc)
    print(ymin)
    print(ymax)
    # plt.axvline(x=xc, ymin=ymin, ymax=ymax, color='k', linestyle='--')
    plt.vlines(x=xc, ymin=ymin, ymax=ymax, color='k')
plt.xlabel('delta', fontsize=18)
plt.ylabel('# clusters', fontsize=18)
plt.show()