import sys

import numpy as np
import math


class Model(object):

    def __init__(self):
        self.layers = []
        self.scores = None

    def add(self, layer):
        self.layers.append(layer)

    def fit(self, X, y, epochs=10, learning_rate=0.01, batch_size=200, validation_fraction=0.1, seed=None, verbose=False):
        self.scores = {'loss': [], 'acc': []}

        if validation_fraction:
            X, y, X_val, y_val = self._validation_split(X, y, validation_fraction)
            self.scores['val_loss'] = []
            self.scores['val_acc'] = []
        else:
            X_val = None
            y_val = None

        n_samples, n_features = X.shape
        random_state = np.random.RandomState(seed)
        self._initialize(n_features, random_state)

        if batch_size is None:
            batch_size = n_samples

        for epoch in range(epochs):
            loss = 0.0
            accuracy = 0.0
            for start in range(0, len(X), batch_size):
                end = start + batch_size
                batch_slice = slice(start, end)

                output, scores = self.evaluate(X[batch_slice], y[batch_slice])
                loss += scores['loss']
                accuracy += scores['acc']

                y_batch = one_hot(y[batch_slice], output.shape[1])
                grads = cross_entropy_gradient(y_batch, output)
                self._backward(grads, learning_rate)

            n_batches = math.ceil(len(X) / batch_size)
            self.scores['loss'].append(loss / n_batches)
            self.scores['acc'].append(accuracy / n_batches)

            if validation_fraction:
                _, scores = self.evaluate(X_val, y_val)
                self.scores['val_loss'].append(scores['loss'])
                self.scores['val_acc'].append(scores['acc'])

            if verbose:
                sys.stdout.write('\rEpoch %*s/%s - loss: %.3f acc: %.3f'
                                 % (len(str(epochs)), epoch + 1, epochs, self.scores['loss'][-1], self.scores['acc'][-1]))
                sys.stdout.flush()

            if verbose:
                sys.stdout.write('\n')

    def predict(self, X):
        return np.argmax(self._forward(X), axis=1)

    def evaluate(self, X, y):
        output = self._forward(X)
        _, n_classes = output.shape
        loss = np.sum(cross_entropy(one_hot(y, n_classes), output)) / len(X)
        accuracy = np.sum(y == np.argmax(output, axis=1)) / len(X)
        return output, {'loss': loss, 'acc': accuracy}

    def _initialize(self, input_shape, random_state):
        for layer in self.layers:
            layer.initialize(input_shape, random_state)
            input_shape = layer.units

    def _validation_split(self, X, y, fraction):
        n_val_samples = math.ceil(fraction * len(X))
        X_val = X[0:n_val_samples]
        y_val = y[0:n_val_samples]
        X = X[n_val_samples:]
        y = y[n_val_samples:]
        return X, y, X_val, y_val

    def _forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def _backward(self, grads, learning_rate):
        for layer in reversed(self.layers):
            weight_grads, bias_grads, grads = layer.backward(grads)
            layer.weights -= learning_rate * weight_grads
            layer.biases -= learning_rate * bias_grads


class Dense(object):

    def __init__(self, units, activation=None):
        self.units = units
        self.activation = activation
        self.input = None
        self.weights = None
        self.biases = None

    def initialize(self, input_shape, random_state):
        self.weights = random_state.normal(0.0, 1.0, size=(input_shape, self.units))
        self.biases = np.zeros(shape=self.units)

    def forward(self, input):
        self.input = input
        Z = np.matmul(self.input, self.weights) + self.biases

        if self.activation is 'softmax':
            return softmax(Z)
        elif self.activation is 'sigmoid':
            return sigmoid(Z)
        else:
            raise ValueError("Unknown activation function '%s'" % self.activation)

    def backward(self, grads):
        Z = np.matmul(self.input, self.weights) + self.biases

        if self.activation is 'softmax':
            jacobian = softmax_gradient(Z)
            grads = np.einsum('ijk,ik->ij', jacobian, grads)
        elif self.activation is 'sigmoid':
            grads *= sigmoid_gradient(Z)
        else:
            raise ValueError('Activation function "%s" is not supported' % self.activation)

        n_samples, _ = grads.shape
        weight_grads = np.matmul(self.input.T, grads) / n_samples
        bias_grads = np.sum(grads, axis=0) / n_samples
        grads = np.matmul(grads, self.weights.T)
        return weight_grads, bias_grads, grads


def softmax(X):
    X = np.copy(X)
    X = X - np.max(X, axis=1, keepdims=True)
    exp_X = np.exp(X)
    return exp_X / np.sum(exp_X, axis=1, keepdims=True)


def softmax_gradient(X):
    smax = softmax(X)
    outer = np.einsum('ij,ik->ijk', smax, smax)
    diagonal = np.einsum('ij,jk->ijk', smax, np.eye(smax.shape[1]))
    jacobian = diagonal - outer
    return jacobian


def sigmoid(X):
    return 1 / (1 + np.exp(-X))


def sigmoid_gradient(X):
    s = sigmoid(X)
    return s * (1 - s)


def cross_entropy(y, p):
    return -y * np.log(p + 1e-12)


def cross_entropy_gradient(y, p):
    return -y / (p + 1e-12)


def one_hot(y, n_classes):
    return np.eye(n_classes)[y]