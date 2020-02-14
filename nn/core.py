import sys
import numpy as np
import math

from nn.callbacks import ScoresCallback, LoggingCallback
from nn.losses import categorical_cross_entropy, cross_entropy_gradient
from nn.utils import to_categorical, split


class Model(object):

    def __init__(self):
        self.layers = []
        self.scorer = ScoresCallback()
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
        callbacks = [self.scorer]
        if verbose:
            callbacks.append(LoggingCallback())

        val_dataset = None
        if validation_fraction:
            (X, y), val_dataset = split(X, y, fraction=validation_fraction)

        if batch_size is None:
            batch_size, _ = X.shape

        for callback in callbacks:
            callback.on_train_begin((X, y), val_dataset)

        for epoch in range(epochs):

            for callback in callbacks:
                callback.on_batch_begin((X, y), val_dataset)

            if verbose:
                sys.stdout.write('Epoch %s/%s\n' % (epoch + 1, epochs))

            if shuffle:
                p = np.random.permutation(len(X))
                X, y = X[p], y[p]

            for start in range(0, len(X), batch_size):
                end = start + batch_size
                batch_slice = slice(start, end)

                output = self._forward(X[batch_slice], training=True)

                loss = self._compute_loss(output, y[batch_slice]) + self._compute_penalty(len(X))

                y_batch = to_categorical(y[batch_slice], output.shape[1])
                grads = cross_entropy_gradient(y_batch, output)
                self._backward(grads)

                for callback in callbacks:
                    callback.on_batch_end(output, loss)

                if verbose:
                    sys.stdout.write('\r%*s/%s - loss: %.4f - accuracy: %.4f' % (len(str(len(X))), start + batch_size, len(X), loss, accuracy))
                    sys.stdout.flush()

            val_output, val_loss = None, None
            if val_dataset is not None:
                val_output = self._forward(X)
                val_loss = self._compute_loss(val_output, y) + self._compute_penalty(len(X))

            for callback in callbacks:
                callback.on_epoch_end(val_output, val_loss)


    def predict(self, X):
        return np.argmax(self._forward(X), axis=1)

    def evaluate(self, X, y, training=False):
        output = self._forward(X, training)
        loss = self._compute_loss(output, y)
        accuracy = self._compute_accuracy(output, y)
        return output, {'loss': loss, 'acc': accuracy}

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
