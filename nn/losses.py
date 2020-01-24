import numpy as np


def categorical_cross_entropy(y, p):
    return -y * np.log(p + 1e-12)


def cross_entropy_gradient(y, p):
    return -y / (p + 1e-12)
