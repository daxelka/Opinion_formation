import numpy as np
import json
import math


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
