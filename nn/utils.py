import numpy as np
import math


def to_categorical(y, n_classes):
    return np.eye(n_classes)[y]


def split(*arrays, fraction):
    left, right = [], []
    for array in arrays:
        boundary = math.ceil(fraction * len(array))
        left.append(array[:-boundary])
        right.append(array[-boundary:])
    return left, right
