import numpy as np
from scipy.integrate import odeint


class Deffuant_MF_canonical:

    def __init__(self, epsilon, p0):
        # find a closest point in grid and make a self.epsilon
        self.epsilon = epsilon
        self.size = len(p0)
        # opinion grid
        self.opinion_grid = np.linspace(0, 1, num=self.size, endpoint=True)
        self.dx = self.opinion_grid[1] - self.opinion_grid[0]
        self.confidence = int(np.rint(self.epsilon / self.dx))  # epsilon translated into number of intervals on
        self.confidence_halved = int(self.confidence / 2)
        self.epsilon_grid = np.linspace(0, self.epsilon, num=self.confidence, endpoint=False)
        self.epsilon_grid_halved = self.epsilon_grid[0:self.confidence_halved]

        self.dz = self.dx * 2
        self.z_size = int(1 / self.dz) + 1
        self.z_grid = np.linspace(0, 1, num=self.z_size, endpoint=True)
        # print('dx:', self.dx, ' x_size:', self.size)
        # print('dz:', self.dz,' z_size:', self.z_size)
        # print('confidence_interval:', int(0.5 / self.dx) + 1)

    # def integrate_inflow(self, p, z_grid, domain_size):
    #     I_inflow = np.empty((domain_size,))
    #     for i in range(0, domain_size):
    #         I_inflow[i] = self.integral_inflow(i, p, z_grid)
    #     return I_inflow

    def integrate_inflow(self, p, x, dx):
        I_inflow = np.empty((len(p),))
        dz = 2 * dx
        for i in range(0, len(p)):
            if x[i] <= 1/2:
                z_N = int(2 * x[i]/dz + 1)
                z_grid = np.linspace(0, 2 * x[i], num=z_N)
                I_inflow[i] = self.integral_inflow(i, p, z_grid)
            else:
                z_N = int(1 / dz + 1)
                z_grid = np.linspace(0, 1, num=z_N)
                I_inflow[i] = self.integral_inflow(i, p, z_grid)
        return I_inflow



    def integral_inflow(self, index, p, z_grid):
        z_size = len(z_grid)
        fun1 = []
        for j in range(0, z_size):
            fun1.append(p[index - j] * p[index + j])

        return np.trapz(fun1, x=z_grid, axis=0)
        # return np.sum(fun1, axis=0) * self.dz

    def integral_outflow(self, index, p, z_grid):
        z_size = len(z_grid)
        fun1 = []
        for j in range(0, z_size):
            fun1.append(p[index] * p[index + 2*j])

        return np.trapz(fun1, x=z_grid, axis=0)

    def integral_inflow_test(self, index, x_grid):
        x = x_grid[index]
        result = 1 / 2 * (np.sin(1) - np.cos(2*x))
        return result

    def integral_outflow_test(self, index, x_grid):
        x = x_grid[index]
        result = np.sin(x) * (-np.cos(x + 1) + np.cos(x))
        return result


