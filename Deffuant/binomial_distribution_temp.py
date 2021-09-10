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
obs_dist = mixture_rvs(prob=[weight1, weight2], size=10,
                        dist=[stats.norm, stats.norm],
                        kwargs = (dict(loc=dist1_loc, scale=dist1_scale),
                                  dict(loc=dist2_loc, scale=dist2_scale)))


# points = [0.1, 0.31,  0.32, 0.45, 0.35, 0.40, 0.5 ]
points = obs_dist

clusters = []
eps = 0.2
points_sorted = sorted(points)
curr_point = points_sorted[0]
curr_cluster = [curr_point]
for point in points_sorted[1:]:
    if point <= curr_point + eps:
        curr_cluster.append(point)
    else:
        clusters.append(curr_cluster)
        curr_cluster = [point]
    curr_point = point
clusters.append(curr_cluster)
print(len(clusters))