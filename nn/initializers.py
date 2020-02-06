import numpy as np
import math


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


class GlorotNormal(object):

    def get_values(self, shape):
        fan_in, fan_out = shape[-2:]
        stddev = math.sqrt(2 / (fan_in + fan_out))
        return np.random.normal(0.0, stddev, size=shape)


class GlorotUniform(object):

    def get_values(self, shape):
        fan_in, fan_out = shape[-2:]
        boundary = math.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-boundary, boundary, size=shape)


class Zeros(object):

    def get_values(self, shape):
        return np.zeros(shape)
