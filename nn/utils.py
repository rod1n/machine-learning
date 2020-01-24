import numpy as np


def to_categorical(y, n_classes):
    return np.eye(n_classes)[y]
