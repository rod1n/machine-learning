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
        fan_in, fan_out = _compute_fans(shape)
        stddev = math.sqrt(2 / (fan_in + fan_out))
        return np.random.normal(0.0, stddev, size=shape)


class GlorotUniform(object):

    def get_values(self, shape):
        fan_in, fan_out = _compute_fans(shape)
        boundary = math.sqrt(6 / (fan_in + fan_out))
        return np.random.uniform(-boundary, boundary, size=shape)


def _compute_fans(shape):
    if len(shape) == 2:
        fan_in, fan_out = shape
    elif len(shape) > 2:
        filters, channels, *kernel = shape
        receptive_field_size = 1
        for dim in kernel:
            receptive_field_size *= dim

        fan_in = filters * receptive_field_size
        fan_out = channels * receptive_field_size
    else:
        raise ValueError('Unexpected shape {}', shape)
    return fan_in, fan_out

class Zeros(object):

    def get_values(self, shape):
        return np.zeros(shape)
