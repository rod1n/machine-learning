class SGD(object):

    def __init__(self, learning_rate=0.1, momentum=0.0, decay=0.0, nesterov=False):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.nesterov = nesterov
        self.decay = decay
        self.iteration = 0

    def update(self, layer, weight_grads, bias_grads):
        if layer.opt_params is None:
            layer.opt_params = {
                'weight_velocity': 0.0,
                'bias_velocity': 0.0
            }
        self.iteration += 1

        layer.opt_params['weight_velocity'] = self._get_velocity(weight_grads, layer.opt_params['weight_velocity'])
        layer.opt_params['bias_velocity'] = self._get_velocity(bias_grads, layer.opt_params['bias_velocity'])

        layer.weights += layer.opt_params['weight_velocity']
        layer.biases += layer.opt_params['bias_velocity']

    def _get_velocity(self, grads, current_velocity):
        learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iteration))

        velocity = self.momentum * current_velocity - learning_rate * grads
        if self.nesterov:
            velocity = self.momentum * velocity - learning_rate * grads
        return velocity
