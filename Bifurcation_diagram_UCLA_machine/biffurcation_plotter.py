import numpy as np
import matplotlib.pyplot as plt
from utils.bifurcation_diagram.plotter import BifurcationDiagramPlotter
import data_tools as mytools

filename = '/Users/daxelka/Research/Deffuant_model/Deffuant-Barabasi/biff_DB_1000_evenly_spaced_repeated_5.txt'

class BifurcationDiagramPlotter:

    def plot(self, x, y, x_label = 'bifurcation parameter', y_label= 'y variable'):
        plt.rcParams['font.size'] = '12'
        plt.scatter(x, y, marker='.', color='black', s=1)
        plt.xlabel(x_label,fontsize=14)
        plt.ylabel(y_label,fontsize=14)
        plt.ylim(-4, 4)
        plt.savefig('/Users/daxelka/Research/Deffuant_model/Deffuant-Barabasi/biff_DB_1000_evenly_spaced_repeated_5.png')
        plt.show()


class BifurcationDiagramGenerator:
    def __init__(self, x_iterator, initial_values_iterator, converge):
        self.x_iterator = x_iterator
        self.initial_values_iterator = initial_values_iterator
        self.converge = converge

    def run(self):
        results = []
        for x in self.x_iterator():
            for initial_value in self.initial_values_iterator():
                converged = self.converge(x, initial_value)  # can be list or np.array ?
                r = list(map(lambda c: [x, c], converged))
                results = results + r

        x_var = [x[0] for x in results]
        y_var = [x[1] for x in results]
        return x_var, y_var


# unpack json
data, x, y, p = mytools.unpack_json(filename)

# transform to delta grid
x_t, y_t = mytools.transform_to_delta_grid(x, y, is_polar=False)

# Plot diagram
BifurcationDiagramPlotter().plot(x_t, y_t, 'Delta', 'opinion')
# BifurcationDiagramPlotter().plot(x, y, 'confidence bound', 'opinion', y_limits=(0,6.5), weight=np.array(p))
# BifurcationDiagramPlotter().plot(x, y, 'confidence bound', 'opinion', y_limits=(0,10), weight=np.array(p))
# show_distribution(data['initial_conditions'][3])
print('here you go')
