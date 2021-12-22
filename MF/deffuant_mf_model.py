import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class Deffuant_MF:

    def __init__(self, epsilon, p0):
        # find a closest point in grid and make a self.epsilon
        self.epsilon = epsilon
        self.size = len(p0)
        self.grid = np.linspace(0, 1, num=self.size, endpoint=True)
        self.dx = self.grid[1] - self.grid[0]
        self.confidence = int(np.rint(self.epsilon / self.dx))
        if self.confidence % 2:  # if odd
            self.confidence_halved = int(self.confidence / 2)
        else:  # if even
            self.confidence_halved = int(self.confidence / 2) - 1

        self.z = np.linspace(self.dx, self.epsilon - self.dx, num=self.confidence - 1, endpoint=True)
        self.z_halved = self.z[0:self.confidence_halved]

        # print(self.dx)

    def equations(self, p, t):
        confidence = int(np.rint(self.epsilon / self.dx))
        confidence_halved = int(np.rint(self.epsilon / self.dx / 2))

        z = np.linspace(self.dx, self.epsilon, num=confidence)
        z_halved = np.linspace(self.dx, self.epsilon, num=confidence_halved)

        dpdt = self.integral_inflow(p, confidence_halved, z_halved) \
               - self.integral_outflow_right(p, confidence, z) \
               - self.integral_outflow_left(p, confidence, z)
        return dpdt

    def integrate(self, p0, t):
        p = odeint(self.equations, p0, t)
        return p, t

    def run(self, p0=None, dt=0.01, T=0.02):
        t = np.linspace(0, T, int(T / dt))
        return self.integrate(p0, t)

    def integral_inflow(self, p, confidence_interval, x):
        # confidence_interval = int(np.rint(self.epsilon / self.dx /2))
        integral = np.empty((len(p),))
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
        # confidence_interval = int(np.rint(self.epsilon / self.dx /2))
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
        # confidence_interval = int(np.rint(self.epsilon / self.dx /2))
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


# initial distribution
# n_nodes = 5
# rng = np.random.default_rng()
# p0= rng.uniform(0, 1, (n_nodes,))
# p0 = np.array([1, 2, 3, 4, 5])

# # p0 = np.array([1,2,3,4,5])
# model = Deffuant_MF(0.5, p0)
# # model.integral2(p0)
# integral = model.integral_inflow(p0, 2, [1, 2])


# model.mask_matrix_epsilon_down()

# result, t = model.run(p0)

# Plotting
# plt.plot(result[-1])
# plt.show()

# def test_integral_inflow():
#     p = np.array([1, 2, 3, 4, 5, 6, 7, 8])
#     x = np.array([1, 2, 3])
#     model = Deffuant_MF(0.5, p)
#     integral = model.integral_inflow(p, 3, x)
#     true_result = np.array([np.trapz(np.array([0, 0, 0]), x=x),
#                             np.trapz(np.array([3, 0, 0]), x=x),
#                             np.trapz(np.array([8, 5, 0]), x=x),
#                             np.trapz(np.array([15, 12, 7]), x=x),
#                             np.trapz(np.array([24, 21, 16]), x=x),
#                             np.trapz(np.array([35, 32, 0]), x=x),
#                             np.trapz(np.array([48, 0, 0]), x=x),
#                             np.trapz(np.array([0, 0, 0]), x=x)])
#     print(np.array_equal(integral, true_result))


# def test_integral_outflow_left():
#     p = np.array([1, 2, 3, 4, 5, 6, 7, 8])
#     x = np.array([1, 2, 3])
#     model = Deffuant_MF(0.5, p)
#     integral = model.integral_outflow_left(p, 3, x)
#     true_result = np.array([np.trapz(np.array([0, 0, 0]), x=x),
#                             np.trapz(np.array([2, 0, 0]), x=x),
#                             np.trapz(np.array([6, 3, 0]), x=x),
#                             np.trapz(np.array([12, 8, 4]), x=x),
#                             np.trapz(np.array([20, 15, 10]), x=x),
#                             np.trapz(np.array([30, 24, 18]), x=x),
#                             np.trapz(np.array([42, 35, 28]), x=x),
#                             np.trapz(np.array([56, 48, 40]), x=x)])
#     print(np.array_equal(integral, true_result))


# def test_integral_outflow_right():
#     p = np.array([1, 2, 3, 4, 5, 6, 7, 8])
#     x = np.array([1, 2, 3])
#     model = Deffuant_MF(0.5, p)
#     integral = model.integral_outflow_right(p, 3, x)
#     true_result = np.array([np.trapz(np.array([2, 3, 4]), x=x),
#                             np.trapz(np.array([6, 8, 10]), x=x),
#                             np.trapz(np.array([12, 15, 18]), x=x),
#                             np.trapz(np.array([20, 24, 28]), x=x),
#                             np.trapz(np.array([30, 35, 40]), x=x),
#                             np.trapz(np.array([42, 48, 0]), x=x),
#                             np.trapz(np.array([56, 0, 0]), x=x),
#                             np.trapz(np.array([0, 0, 0]), x=x)])
#     print(np.array_equal(integral, true_result))


# test_integral_inflow()
# test_integral_outflow_left()
# test_integral_outflow_right()

# Testing intervals
# p = np.linspace(0, 1, 22)
# print(p)
# model = Deffuant_MF(0.5, p)
# p,t = model.integrate(p0=p0,t=[1,2])
# plt.plot(t, p)
# plt.show()
