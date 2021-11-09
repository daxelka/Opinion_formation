import numpy as np
import json
from utils.bifurcation_diagram.plotter import BifurcationDiagramPlotter

filename = '/Users/daxelka/Research/Deffuant_model/ABM_simulation/data/cos_2peaks_10k.txt'

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


y_t = (np.array(y) - 0.5)/np.array(x)
x_t = 0.5/np.array(x)

BifurcationDiagramPlotter().plot(x_t, y_t, 'confidence bound', 'opinion', y_limits=(-5,5), weight=np.array(p))
# BifurcationDiagramPlotter().plot(x, y, 'confidence bound', 'opinion', weight=np.array(p))