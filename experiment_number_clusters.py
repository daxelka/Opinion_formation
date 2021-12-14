import numpy as np
from utils.bifurcation_diagram.plotter import BifurcationDiagramPlotter
import math
import data_tools as mytools

filename = '/Users/daxelka/Research/Deffuant_model/ABM_simulation/data/polar_bifurcation_10k_20runs.txt'
# filename = '/Users/daxelka/Research/Deffuant_model/ABM_simulation/data/uniform_10k_20runs.txt'

data_processed = {}
def unpack_json_filtered(filename):
    with open(filename) as json_file:
        data = mytools.json.load(json_file)
        for parameter, results in data['experiments'].items():
            for r in results:
                means = []
                for c in r:
                    if c[1] > 0.7 * float(parameter) / 0.5:
                        means.append(c[0])
                    if len(means) > 1:
                        distances = np.diff(np.array(means)/ math.pi / 2).tolist()
                        # distances = np.diff(np.array(means)).tolist()
                    else:
                        distances = [0.5]
                    if data_processed.get(parameter) is None:
                        data_processed[parameter] = []
                    else:
                        if distances:
                            data_processed[parameter].append([distances])

    return data_processed, data


# unpack json
data_processed, data = unpack_json_filtered(filename)

y = []
for parameter, results in data_processed.items():
    for r in results:
        for c in r:
            y_i = list(map(lambda n: [float(parameter), n], c))
            y = y + y_i
            print(c)

x_var = [x[0] for x in y]
y_var = [x[1] for x in y]

# Plot diagram
BifurcationDiagramPlotter().plot(x_var, y_var, 'confidence bound', '# clusters', title= 'classical Deffuant model')
print('here you go')
