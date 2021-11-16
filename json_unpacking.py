import numpy as np
import json
from utils.bifurcation_diagram.plotter import BifurcationDiagramPlotter
import math
from distribution_tools import show_distribution

filename = '/Users/daxelka/Research/Deffuant_model/ABM_simulation/data/experiment_polar_single_IC_10_runs.txt'

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


y_t = (np.array(y)/2/math.pi - 0.5)/np.array(x)
# y_t = (np.array(y) - 0.5)/np.array(x)
x_t = 0.5/np.array(x)

# BifurcationDiagramPlotter().plot(x_t, y_t, 'confidence bound', 'opinion', y_limits=(-5,5), weight=np.array(p))
BifurcationDiagramPlotter().plot(x, y, 'confidence bound', 'opinion', y_limits=(0,7), weight=np.array(p))
# show_distribution(data['initial_conditions'][3])