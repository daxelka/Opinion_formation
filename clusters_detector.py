# from numpy import array, linspace
# from sklearn.neighbors.kde import KernelDensity
# from matplotlib.pyplot import plot
#
# a = array([10,11,9,23,21,11,45,20,11,12]).reshape(-1, 1)
# kde = KernelDensity(kernel='gaussian', bandwidth=3).fit(a)
# s = linspace(0,50)
# e = kde.score_samples(s.reshape(-1,1))
# plot(s, e)

# import numpy as np
# from scipy.signal import argrelextrema
# x = np.array([2, 1, 2, 3, 2, 0, 1, 0])
# print(argrelextrema(x, np.greater))

import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.distributions.mixture_rvs import mixture_rvs
from scipy.signal import argrelextrema

# Location, scale and weight for the two distributions
dist1_loc, dist1_scale, weight1 = -1 , .5, .25
dist2_loc, dist2_scale, weight2 = 1 , .5, .75

# Sample from a mixture of distributions
obs_dist = mixture_rvs(prob=[weight1, weight2], size=100,
                        dist=[stats.norm, stats.norm],
                        kwargs = (dict(loc=dist1_loc, scale=dist1_scale),
                                  dict(loc=dist2_loc, scale=dist2_scale)))

# fig = plt.figure(figsize=(12, 5))
# ax = fig.add_subplot(111)
#
# # Scatter plot of data samples and histogram
# ax.scatter(obs_dist, np.abs(np.random.randn(obs_dist.size)),
#             zorder=15, color='red', marker='x', alpha=0.5, label='Samples')
# lines = ax.hist(obs_dist, bins=20, edgecolor='k', label='Histogram')
#
# ax.legend(loc='best')
# ax.grid(True, zorder=-5)
# plt.show()

# Fitting with the default arguments
kde = sm.nonparametric.KDEUnivariate(obs_dist)
kde.fit()  # Estimate the densities

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(111)

# Plot the histrogram
ax.hist(obs_dist, bins=20, density=True, label='Histogram from samples',
        zorder=5, edgecolor='k', alpha=0.5)

# Plot the KDE as fitted using the default arguments
ax.plot(kde.support, kde.density, lw=3, label='KDE from samples', zorder=10)
plt.show()

# Finding extrema
mi, ma = argrelextrema(kde.density, np.less)[0], argrelextrema(kde.density, np.greater)[0]

print(mi)
print(ma)

a = obs_dist.reshape(-1, 1)
print(a.shape)
print('Grouping by HAS QUIT:')
print(a[a < mi[0]], a[(a >= mi[0]) * (a <= mi[1])], a[a >= mi[1]])
# print('Grouping by yasirroni:')
# print(a[a < s[mi][0]], a[(a >= s[mi][0]) * (a <= s[mi][1])], a[a >= s[mi][1]])

# points = [0.1, 0.31,  0.32, 0.45, 0.35, 0.40, 0.5 ]
#
# clusters = []
# eps = 0.2
# points_sorted = sorted(points)
# curr_point = points_sorted[0]
# curr_cluster = [curr_point]
# for point in points_sorted[1:]:
#     if point <= curr_point + eps:
#         curr_cluster.append(point)
#     else:
#         clusters.append(curr_cluster)
#         curr_cluster = [point]
#     curr_point = point
# clusters.append(curr_cluster)
# print(clusters)˜