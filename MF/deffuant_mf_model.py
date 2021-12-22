import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import time
import distribution_tools as tools


class Deffuant_MF:

    def __init__(self, epsilon, p0):
        # find a closest point in grid and make a self.epsilon
        self.epsilon = epsilon
        self.size = len(p0)
        # opinion grid
        self.opinion_grid = np.linspace(0, 1, num=self.size, endpoint=True)
        self.dx = self.opinion_grid[1] - self.opinion_grid[0]
        self.confidence = int(np.rint(self.epsilon / self.dx))  # epsilon translated into number of intervals on
        # self.grid
        if self.confidence % 2:  # if odd
            self.confidence_halved = int(self.confidence / 2)
        else:  # if even
            self.confidence_halved = int(self.confidence / 2) - 1
        # epsilon grid
        # self.epsilon_grid = np.linspace(self.dx, self.epsilon - self.dx, num=self.confidence - 1, endpoint=True)
        self.epsilon_grid = np.linspace(0, self.epsilon, num=self.confidence, endpoint=False)
        self.epsilon_grid_halved = self.epsilon_grid[0:self.confidence_halved]
        print(self.confidence)
        print(self.confidence_halved)
        print(self.epsilon_grid.shape)
        print(self.epsilon_grid_halved.shape)
        # print(self.epsilon_grid)

        # print(self.dx)

    def equations(self, p, t):
        # confidence = int(np.rint(self.epsilon / self.dx))
        # confidence_halved = int(np.rint(self.epsilon / self.dx / 2))
        #
        # z = np.linspace(self.dx, self.epsilon, num=confidence)
        # z_halved = np.linspace(self.dx, self.epsilon, num=confidence_halved)

        dpdt = 2 * self.integral_inflow(p, self.confidence_halved, self.epsilon_grid_halved) \
               - self.integral_outflow_right(p, self.confidence, self.epsilon_grid) \
               - self.integral_outflow_left(p, self.confidence, self.epsilon_grid)
        return dpdt

    def integrate(self, p0, t):
        p = odeint(self.equations, p0, t)
        return p, t

    def run(self, p0=None, dt=0.01, T=0.02):
        t = np.linspace(0, T, int(T / dt))
        return self.integrate(p0, t)

    def integral_inflow(self, p, confidence_interval, x):
        integral = np.empty((len(p),))
        # extend p with zeros to the left nad the right
        p_extended = np.concatenate((np.zeros((confidence_interval,)),
                                     p,
                                     np.zeros((confidence_interval,))))
        for i in range(len(p)):
            fun = []
            for j in range(1, confidence_interval + 1):
                fun.append(p_extended[i + confidence_interval - j] * p_extended[i + confidence_interval + j])

            integral[i] = np.trapz(fun, x=x, axis=0)
        return integral

    def integral_outflow_left(self, p, confidence_interval, x):
        integral = np.empty((len(p),))
        p_extended = np.concatenate((np.zeros((confidence_interval,)),
                                     p,
                                     np.zeros((confidence_interval,))))
        for i in range(len(p)):
            fun = []
            for j in range(1, confidence_interval + 1):
                fun.append(p_extended[i + confidence_interval] * p_extended[i + confidence_interval - j])

            integral[i] = np.trapz(fun, x=x, axis=0)
        return integral

    def integral_outflow_right(self, p, confidence_interval, x):
        integral = np.empty((len(p),))
        p_extended = np.concatenate((np.zeros((confidence_interval,)),
                                     p,
                                     np.zeros((confidence_interval,))))
        for i in range(len(p)):
            fun = []
            for j in range(1, confidence_interval + 1):
                fun.append(p_extended[i + confidence_interval] * p_extended[i + confidence_interval + j])

            integral[i] = np.trapz(fun, x=x, axis=0)

        return integral


    # def integral_outflow_left_matrix(self, p, confidence_interval, x):
    #     M1 = np.tile(np.array([p]).transpose(), (1, self.size))
    #     M2 = np.tile(p, (self.size, 1))
    #     # M2 = np.tril(np.tile(p, (self.size,1)), k=-1)
    #     mask = self.mask_matrix_epsilon_down(confidence_interval)
    #     # Leave only n elements in M
    #     print(M1)
    #     print(M2)
    #     print(mask)
    #     M = M1 * M2 * mask
    #     print(M)
    #     # integral = np.trapz(M, x=self.grid, axis=1)
    #     # integral_trapz = np.trapz(M, x=x, axis=1)
    #     # print(integral_trapz
    #     dx = x[1] - x[0]
    #     integral = np.sum(M, axis= 1) * dx
    #     return integral
    #
    # def integral2(self, p):
    #     M2 = np.tile(p, (self.size, 1))
    #     print(M2)
    #     # result = np.diagonal(M2, 0)
    #     result_plus = []
    #     result_minus = []
    #     for i in range(1, 3):
    #         diagonal_plus = np.diagonal(M2, i)
    #         diagonal_minus = np.diagonal(M2, (-1) * i)
    #         pad_width = self.size - len(diagonal_plus)
    #         s_plus = np.pad(diagonal_plus, (0, pad_width), 'constant', constant_values=(0, 0))
    #         s_minus = np.pad(diagonal_minus, (pad_width, 0), 'constant', constant_values=(0, 0))
    #         result_plus.append(s_plus)
    #         result_minus.append(s_minus)
    #         print(np.transpose(result_plus))
    #         print(np.transpose(result_minus))
    #         print(np.transpose(result_plus) * np.transpose(result_minus))
    #     return np.transpose(result_plus)
    #
    # def mask_matrix_epsilon_down(self, confidence_interval):
    #     # confidence_interval = int(np.rint(self.epsilon / self.dx))
    #     a = []
    #     for i in range(1, confidence_interval + 1):
    #         a.append(np.diag(np.ones(self.size - i), k=-i))
    #     return np.sum(np.array(a), axis=0)



# Initial condition
N_nodes = 1000
p0 = tools.uniform_opinion(N_nodes, (0,1))

# model initialisation
model = Deffuant_MF(0.3, p0)

# integrating the model
# p,t = model.integrate(p0=p0, t=np.linspace(0.01, 10, 100))

t0 = time.perf_counter()

p,t = model.run(p0=p0, dt=0.01, T=150)

t1 = time.perf_counter()
print('performance time:', t1 - t0)

# plotting
plt.plot(model.opinion_grid, p[-1, :])
plt.show()

# print(model.equations(p0,1))
# print(model.integral_outflow_left(p0, model.confidence, model.epsilon_grid))