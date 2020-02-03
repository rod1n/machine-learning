import numpy as np


class SGD(object):

    def __init__(self, learning_rate=0.1, momentum=0.0, decay=0.0, nesterov=False):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.decay = decay
        self.iteration = 0

    def update(self, layer, weight_grads, bias_grads):
        self.iteration += 1
        self._update_variable(layer, 'weights', weight_grads)
        self._update_variable(layer, 'biases', bias_grads)

    def _update_variable(self, layer, name, grads):
        learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iteration))

        if name not in layer.opt_params:
            layer.opt_params[name] = {'velocity': 0.0}

        velocity = layer.opt_params[name]['velocity']
        velocity = self.momentum * velocity - learning_rate * grads
        if self.nesterov:
            velocity = self.momentum * velocity - learning_rate * grads

        layer.opt_params[name]['velocity'] = velocity

        variable = getattr(layer, name)
        variable += velocity


class Adagrad(object):

    def __init__(self, learning_rate=0.1, decay=0.0):
        self.learning_rate = learning_rate
        self.decay = decay
        self.iteration = 0

    def update(self, layer, weight_grads, bias_grads):
        self.iteration += 1
        self._update_variable(layer, 'weights', weight_grads)
        self._update_variable(layer, 'biases', bias_grads)

    def _update_variable(self, layer, name, grads):
        if name not in layer.opt_params:
            layer.opt_params[name] = {'accumulated_grads': 1e-12}

        layer.opt_params[name]['accumulated_grads'] += grads ** 2
        learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iteration))
        adaptive_learning_rate = learning_rate / np.sqrt(layer.opt_params[name]['accumulated_grads'])

        variable = getattr(layer, name)
        variable -= adaptive_learning_rate * grads


class RMSProp(object):

    def __init__(self, learning_rate=0.01, rho=0.9, decay=0.0):
        self.learning_rate = learning_rate
        self.rho = rho
        self.decay = decay
        self.iteration = 0

    def update(self, layer, weight_grads, bias_grads):
        self.iteration += 1
        self._update_variable(layer, 'weights', weight_grads)
        self._update_variable(layer, 'biases', bias_grads)

    def _update_variable(self, layer, name, grads):
        if name not in layer.opt_params:
            layer.opt_params[name] = {'accumulated_grads': 1e-12}

        accumulated_grads = layer.opt_params[name]['accumulated_grads']
        accumulated_grads = self.rho * accumulated_grads + (1 - self.rho) * grads ** 2

        learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iteration))
        adaptive_learning_rate = learning_rate / np.sqrt(accumulated_grads)

        layer.opt_params[name]['accumulated_grads'] = accumulated_grads

        variable = getattr(layer, name)
        variable -= adaptive_learning_rate * grads


class Adam(object):

    def __init__(self, learning_rate=0.01, beta1=0.9, beta2=0.9, decay=0.0):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.decay = decay
        self.iteration = 0

    def update(self, layer, weight_grads, bias_grads):
        self.iteration += 1
        self._update_variable(layer, 'weights', weight_grads)
        self._update_variable(layer, 'biases', bias_grads)

    def _update_variable(self, layer, name, grads):
        if name not in layer.opt_params:
            layer.opt_params[name] = {'velocity': 0.0, 'accumulated_grads': 1e-12}

        velocity = layer.opt_params[name]['velocity']
        accumulated_grads = layer.opt_params[name]['accumulated_grads']

        velocity = self.beta1 * velocity + (1 - self.beta1) * grads
        accumulated_grads = self.beta2 * accumulated_grads + (1 - self.beta2) * grads ** 2

        learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iteration))
        adaptive_learning_rate = learning_rate / np.sqrt(accumulated_grads)

        layer.opt_params[name]['velocity'] = velocity
        layer.opt_params[name]['accumulated_grads'] = accumulated_grads

        variable = getattr(layer, name)
        variable -= adaptive_learning_rate * velocity
