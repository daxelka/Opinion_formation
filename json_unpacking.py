import numpy as np
from utils.bifurcation_diagram.plotter import BifurcationDiagramPlotter
import data_tools as mytools

filename = '/Users/daxelka/Research/Deffuant_model/ABM_simulation/data/cos_3peaks_0.1ampl.txt'

# unpack json
data, x, y, p = mytools.unpack_json(filename)

# trasfrom
x_t, y_t = mytools.transform_to_delta_grid(x, y)


# BifurcationDiagramPlotter().plot(x_t, y_t, 'confidence bound', 'opinion', y_limits=(-5,5), weight=np.array(p))
BifurcationDiagramPlotter().plot(x, y, 'confidence bound', 'opinion', y_limits=(0,7), weight=np.array(p))
# show_distribution(data['initial_conditions'][3])
print('here you go')