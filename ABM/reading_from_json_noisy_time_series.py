import json
import math
import random
import numpy as np
import matplotlib.pyplot as plt

# Loading results and reproducing noisy time series

filename = '/Users/daxelka/Research/Deffuant_model/results_txt/noisy_circular_e0.316_g0.4_m0.01.json'
with open(filename, 'r') as f:
    d = json.load(f)

N_sample = 100
# Sample N_sample nodes for illustration
opinion_sample = random.sample(range(d['N_nodes']), N_sample)
opinions_np = np.array(d['opinions'])
opinions_selected = []
opinions_selected = opinions_np[:, opinion_sample]

# Creating time list
time_variable = []
[time_variable.append(list(np.ones(N_sample) * time)) for time in d['time']]


print('opinions selected')

plt.scatter(time_variable, opinions_selected, s=0.1, c='black')
# plt.ylim(0, 1)
plt.ylim(0, 2*math.pi)
plt.xlim(d['start_point_recording'], d['n_steps'])
plt.title(r'$\varepsilon:$' + ' ' + str(d['epsilon']) + ', '
          + r'$\gamma:$' + ' ' + str(d['gamma']) + ', '
          + r'$m:$' + ' ' + str(d['m']) + ', '
          + r'$N_{nodes}:$' + ' ' + str(d['N_nodes']),
          fontsize=20)
plt.xlabel('t (MCS)', fontsize=20)
plt.ylabel('opinion', fontsize=20)
plt.xticks(fontsize=14)
plt.show()