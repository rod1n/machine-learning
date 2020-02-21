import sys
import numpy as np
import math
from nn.losses import categorical_cross_entropy, cross_entropy_gradient
from nn.utils import to_categorical, split


class Model(object):

    def __init__(self):
        self.layers = []
        self.scores = None
        self.optimizer = None

    def add(self, layer):
        self.layers.append(layer)

    def compile(self, optimizer):
        self.optimizer = optimizer
        input_shape = None
        for layer in self.layers:
            layer.build(input_shape)
            input_shape = layer.output_shape

    def fit(self, X, y, epochs=10, batch_size=200, validation_fraction=0, shuffle=True, verbose=False):
        self.scores = {'loss': [], 'acc': [], 'val_loss': [], 'val_acc': []}

        val_data = None
        if validation_fraction:
            (X, y), val_data = split(X, y, fraction=validation_fraction)

        if batch_size is None:
            batch_size, _ = X.shape

        for epoch in range(epochs):
            if verbose:
                sys.stdout.write('Epoch %s/%s\n' % (epoch + 1, epochs))

            if shuffle:
                p = np.random.permutation(len(X))
                X, y = X[p], y[p]

            total_loss = 0.0
            total_accuracy = 0.0
            for start in range(0, len(X), batch_size):
                end = start + batch_size
                batch_slice = slice(start, end)

                output = self._forward(X[batch_slice], training=True)

                loss = self._compute_loss(output, y[batch_slice]) + self._compute_penalty(len(X))
                accuracy = self._compute_accuracy(output, y[batch_slice])

                total_loss += loss
                total_accuracy += accuracy

                y_batch = to_categorical(y[batch_slice], output.shape[1])
                grads = cross_entropy_gradient(y_batch, output)
                self._backward(grads)

                if verbose:
                    sys.stdout.write('\r%*s/%s - loss: %.4f - accuracy: %.4f' % (len(str(len(X))), start + batch_size, len(X), loss, accuracy))
                    sys.stdout.flush()

            n_batches = math.ceil(len(X) / batch_size)
            self.scores['loss'].append(total_loss / n_batches)
            self.scores['acc'].append(total_accuracy / n_batches)

            if val_data:
                self._validate(*val_data, self._compute_penalty(len(X)))

            if verbose:
                sys.stdout.write('\r%s/%s - loss: %.4f - accuracy: %.4f'
                                 % (len(X), len(X), self.scores['loss'][-1], self.scores['acc'][-1]))
                sys.stdout.write('\n')
                sys.stdout.flush()

    def predict(self, X):
        return np.argmax(self._forward(X), axis=1)

    def evaluate(self, X, y, training=False):
        output = self._forward(X, training)
        loss = self._compute_loss(output, y)
        accuracy = self._compute_accuracy(output, y)
        return output, {'loss': loss, 'acc': accuracy}

    def _validate(self, X, y, penalty):
        val_output = self._forward(X)
        self.scores['val_loss'].append(self._compute_loss(val_output, y) + penalty)
        self.scores['val_acc'].append(self._compute_accuracy(val_output, y))

    def _compute_loss(self, output, y):
        n_samples, n_classes = output.shape
        return np.sum(categorical_cross_entropy(to_categorical(y, n_classes), output)) / n_samples

    def _compute_accuracy(self, output, y):
        n_samples, _ = output.shape
        return np.sum(y == np.argmax(output, axis=1)) / n_samples

    def _compute_penalty(self, m):
        penalty = 0.0
        for layer in self.layers:
            if hasattr(layer, 'regularizer') and layer.regularizer is not None:
                penalty += layer.regularizer.get_penalty(layer.weights) / m
        return penalty

    def _forward(self, X, training=False):
        output = X
        for layer in self.layers:
            output = layer.forward(output, training=training)
        return output

    def _backward(self, grads):
        n_samples, _ = grads.shape

        for layer in reversed(self.layers):
            weight_grads, bias_grads, grads = layer.backward(grads)

            if hasattr(layer, 'weights') and hasattr(layer, 'biases'):
                if hasattr(layer, 'regularizer') and layer.regularizer is not None:
                    weight_grads += layer.regularizer.get_grads(layer.weights) / n_samples

                self.optimizer.update(layer, weight_grads, bias_grads)
