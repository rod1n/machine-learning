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


class L1L2(object):

    def __init__(self, l1, l2):
        self.l1 = L1(l1)
        self.l2 = L2(l2)

    def get_penalty(self, weights):
        return self.l1.get_penalty(weights) + self.l2.get_penalty(weights)

    def get_grads(self, weights):
        return self.l1.get_grads(weights) + self.l2.get_grads(weights)
