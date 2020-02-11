import numpy as np


class L1(object):

    def __init__(self, l):
        self.l = l

    def get_penalty(self, weights):
        return self.l * np.sum(np.abs(weights))

    def get_grads(self, weights):
        return self.l * np.sign(weights)


class L2(object):

    def __init__(self, l):
        self.l = l

    def get_penalty(self, weights):
        return self.l * np.sum(np.square(weights))

    def get_grads(self, weights):
        return self.l * weights / 2
