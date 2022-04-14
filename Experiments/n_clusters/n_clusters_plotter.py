import numpy as np
import matplotlib.pyplot as plt
from utils.bifurcation_diagram.plotter import BifurcationDiagramPlotter
import data_tools as mytools

circular = '/Users/daxelka/Research/Deffuant_model/Simulations/data/polar_bifurcation_10k_20runs.txt'
standart = '/Users/daxelka/Research/Deffuant_model/Simulations/data/uniform_10k_20runs.txt'

# unpack json
_, x_c, y_c, _ = mytools.unpack_n_clusters(standart)
_, x_s, y_s, _ = mytools.unpack_n_clusters(circular)

# Plot diagram
# BifurcationDiagramPlotter().plot(1/np.array(x_c)/2, y_c, 'confidence bound', '# clusters', title= 'classical Deffuant model', y_limits=(0,6))
# print('here you go')
plt.scatter(1/np.array(x_c)/2, y_c, marker='.', color='black', s=10)
plt.scatter(1/np.array(x_s)/2, np.array(y_s)+0.1, marker='.', color='blue', s=10)
plt.show()