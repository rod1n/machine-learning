import numpy as np


class RandomNormal(object):

    def __init__(self, mean=0.0, stddev=1.0):
        self.mean = mean
        self.stddev = stddev

    def get_values(self, shape):
        return np.random.normal(self.mean, self.stddev, size=shape)


class RandomUniform(object):

    def __init__(self, low=0.0, high=1.0):
        self.low = low
        self.high = high

    def get_values(self, shape):
        return np.random.uniform(self.low, self.high, size=shape)
