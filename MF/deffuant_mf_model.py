import numpy as np
from scipy.integrate import odeint


class Deffuant_MF:

    def __init__(self, epsilon, p0):
        # find a closest point in grid and make a self.epsilon
        self.epsilon = epsilon
        self.size = len(p0)
        # opinion grid
        self.opinion_grid = np.linspace(0, 1, num=self.size, endpoint=True)
        self.dx = self.opinion_grid[1] - self.opinion_grid[0]
        self.confidence = int(np.rint(self.epsilon / self.dx))  # epsilon translated into number of intervals on
        self.confidence_halved = int(self.confidence / 2)
        # if self.confidence % 2:  # if odd
        #     self.confidence_halved = int(self.confidence / 2)
        # else:  # if even
        #     self.confidence_halved = int(self.confidence / 2) - 1
        # epsilon grid
        # self.epsilon_grid = np.linspace(self.dx, self.epsilon - self.dx, num=self.confidence - 1, endpoint=True)
        self.epsilon_grid = np.linspace(0, self.epsilon, num=self.confidence, endpoint=False)
        self.epsilon_grid_halved = self.epsilon_grid[0:self.confidence_halved]
        print(self.confidence)
        print(self.confidence_halved)
        print(self.epsilon_grid.shape)
        print(self.epsilon_grid_halved.shape)

    def equations(self, p, t):
        dpdt =  2 * self.integral_inflow(p, self.confidence_halved, self.epsilon_grid_halved) \
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
        extension_interval = confidence_interval + 1
        p_extended = np.concatenate((np.zeros((extension_interval,)),
                                     p,
                                     np.zeros((extension_interval,))))
        for i in range(len(p)):
            fun1 = []
            fun2 = []
            fun3 = []
            for j in range(1, confidence_interval + 1):
                fun1.append(p_extended[i + extension_interval - j] * p_extended[i + extension_interval + j])

            for j in range(0, confidence_interval + 1):
                fun2.append(p_extended[i + extension_interval - j] * p_extended[i + extension_interval + j + 1])
                fun3.append(p_extended[i + extension_interval - j - 1] * p_extended[i + extension_interval + j])

            # integral[i] = np.trapz(fun1, x=x, axis=0) + 1/2 * np.trapz(fun2, x=x, axis=0) + 1/2 * np.trapz(fun3, x=x, axis=0)
            integral[i] = self.dx * (np.sum(fun1, axis=0) + 1 / 2 * np.trapz(fun2, axis=0) + 1 / 2 * np.trapz(fun3, axis=0))
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

    def total_mass(self, p):
        return np.trapz(p, self.opinion_grid)
