import unittest
import numpy as np
from deffuant_mf_model import Deffuant_MF


class TestMFIntegrals(unittest.TestCase):

    def test_integral_outflow_left(self):
        p = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        x = np.array([1, 2, 3])
        model = Deffuant_MF(0.5, p)
        integral = model.integral_outflow_left(p, 3, x)
        true_result = np.array([np.trapz(np.array([0, 0, 0]), x=x),
                                np.trapz(np.array([2, 0, 0]), x=x),
                                np.trapz(np.array([6, 3, 0]), x=x),
                                np.trapz(np.array([12, 8, 4]), x=x),
                                np.trapz(np.array([20, 15, 10]), x=x),
                                np.trapz(np.array([30, 24, 18]), x=x),
                                np.trapz(np.array([42, 35, 28]), x=x),
                                np.trapz(np.array([56, 48, 40]), x=x)])
        self.assertTrue(np.array_equal(integral, true_result))

    def test_integral_outflow_right(self):
        p = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        x = np.array([1, 2, 3])
        model = Deffuant_MF(0.5, p)
        integral = model.integral_outflow_right(p, 3, x)
        true_result = np.array([np.trapz(np.array([2, 3, 4]), x=x),
                                np.trapz(np.array([6, 8, 10]), x=x),
                                np.trapz(np.array([12, 15, 18]), x=x),
                                np.trapz(np.array([20, 24, 28]), x=x),
                                np.trapz(np.array([30, 35, 40]), x=x),
                                np.trapz(np.array([42, 48, 0]), x=x),
                                np.trapz(np.array([56, 0, 0]), x=x),
                                np.trapz(np.array([0, 0, 0]), x=x)])
        self.assertTrue(np.array_equal(integral, true_result))

    def test_integral_inflow(self):
        p = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        x = np.array([1, 2, 3])
        model = Deffuant_MF(0.5, p)
        integral = model.integral_inflow(p, 3, x)
        true_result = np.array([np.trapz(np.array([0, 0, 0]), x=x),
                                np.trapz(np.array([3, 0, 0]), x=x),
                                np.trapz(np.array([8, 5, 0]), x=x),
                                np.trapz(np.array([15, 12, 7]), x=x),
                                np.trapz(np.array([24, 21, 16]), x=x),
                                np.trapz(np.array([35, 32, 0]), x=x),
                                np.trapz(np.array([48, 0, 0]), x=x),
                                np.trapz(np.array([0, 0, 0]), x=x)])
        self.assertTrue(np.array_equal(integral, true_result))


if __name__ == '__main__':
    unittest.main()

