import numpy as np
import matplotlib.pyplot as plt
import time
from deffuant_mf_model import Deffuant_MF


# Initial condition
N_nodes = 1001
delta = 5
# confidence_bound = 0.2
confidence_bound = 1/2/delta
p0 = np.ones(N_nodes)

# time scale
time_dimensionless = 10
time_steps = int(time_dimensionless / confidence_bound)
print('time steps required: ', time_steps)

# model initialisation
model = Deffuant_MF(confidence_bound, p0)

# Model integrating
t0 = time.perf_counter()
print('integrating...')
# p, t = model.run(p0=p0, dt=1, T=time_steps)
p_t_m1 = p0
t_m1 = 0
dt = 0.02
for i in range(100):
    p, t = model.run(p0=p_t_m1, dt=dt, T=dt)
    p_t_m1 = p[-1, :]
    t_m1 = t
    print(model.total_mass(p[-1, :]))

t1 = time.perf_counter()
print('done, performance time:', t1 - t0)

# plotting
plt.plot(model.opinion_grid, p[-1, :])
plt.show()

# print(model.total_mass(p[-1,:]))
