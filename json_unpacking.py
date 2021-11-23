import numpy as np
from utils.bifurcation_diagram.plotter import BifurcationDiagramPlotter
import data_tools as mytools

filename = '/Users/daxelka/Research/Deffuant_model/ABM_simulation/data/polar.txt'

# unpack json
data, x, y, p = mytools.unpack_json(filename)

# transform to delta grid
x_t, y_t = mytools.transform_to_delta_grid(x, y, is_polar=True)

# Plot diagram
BifurcationDiagramPlotter().plot(x_t, y_t, 'Delta', 'opinion', y_limits=(-5,5), weight=np.array(p))
# BifurcationDiagramPlotter().plot(x, y, 'confidence bound', 'opinion', y_limits=(0,1), weight=np.array(p))
# show_distribution(data['initial_conditions'][3])
print('here you go')
