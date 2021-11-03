import numpy as np
import json
from utils.bifurcation_diagram.plotter import BifurcationDiagramPlotter

x = []
y = []
p = []

with open('/Users/daxelka/Research/Deffuant_model/ABM_simulation/data/data.txt') as json_file:
    data = json.load(json_file)
    for parameter, results in data['experiments'].items():
        for r in results:
            for c in r:
                x.append(float(parameter))
                y.append(c[0])
                p.append(c[1])


y_t = (np.array(y) - 0.5)/np.array(x)
x_t = 0.5/np.array(x)

BifurcationDiagramPlotter().plot(x_t, y_t, np.array(p), 'confidence bound', 'opinion', y_limits=(-5,5))