import numpy as np
import matplotlib.pyplot as plt
import time
import math
from deffuant_mf_canonical import Deffuant_MF_canonical


# Initial condition
dx = 0.1
# domain limits
x_min, x_max = 0, 4

N_nodes = int((x_max - x_min)/dx + 1)
print('N_nodes:', N_nodes)
# delta = 5
# # confidence_bound = 0.2
# confidence_bound = 1/2/delta
# p0 = np.ones(N_nodes)
#
# # Create x space
x = np.linspace(x_min, x_max, num=N_nodes)
#
#
# # Initial P
p0 = np.sin(x)

# Create z space
# dz must be twice bigger than dx
dz = 2 * dx
z_N = int(1/dz + 1)
print('z_N:',z_N)
z = np.linspace(0, 1, num=z_N)


plt.plot(x, p0)
plt.show()
#
# # model initialisation
model = Deffuant_MF_canonical(1, p0)
#
index = int(N_nodes/2)
print("index:", index, 'x_i:', x[index], 'p_i:', p0[index])

int_true = model.integral_inflow_test(index, x_grid=x)
int_num = model.integral_inflow(index, p0, z)
print(int_true)
print(int_num)

int2_true = model.integral_outflow_test(index, x_grid=x)
int2_num = model.integral_outflow(index, p0, z)
print(int2_true)
print(int2_num)

# I_inflow = np.empty((5,))
# for i in range(0, 5):
#     I_inflow[i] = model.integral_inflow(index + i, p0, z)
# print(I_inflow)

I_inflow = model.integrate_inflow(p0, x, dx)


