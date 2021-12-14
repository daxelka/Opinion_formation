import numpy as np
from utils.bifurcation_diagram.plotter import BifurcationDiagramPlotter
import data_tools as mytools

# filename = '/Users/daxelka/Research/Deffuant_model/ABM_simulation/data/polar_bifurcation_10k_20runs.txt'
filename = '/Users/daxelka/Research/Deffuant_model/ABM_simulation/data/uniform_10k_20runs.txt'

# unpack json
data, x, y, p = mytools.unpack_n_clusters(filename)

# Plot diagram
BifurcationDiagramPlotter().plot(x, y, 'confidence bound', '# clusters', title= 'classical Deffuant model', y_limits=(0,6))
print('here you go')
