from nn import convolve
import numpy as np


def test_convolution():
    input = np.array([[[0, 1, 2],
                      [1, 2, 3],
                      [2, 1, 0]]])
    filter = np.array([[[1, 2],
                        [2, 1]]])
    actual = convolve(input, filter)
    expected = np.array([[[[6, 12],
                           [10, 10]]]])

    np.testing.assert_array_equal(actual, expected)
