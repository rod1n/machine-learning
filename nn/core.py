import sys
import numpy as np
import math
from nn.losses import categorical_cross_entropy, cross_entropy_gradient
from nn.utils import to_categorical


class Model(object):

    def __init__(self):
        self.layers = []
        self.scores = None

    def add(self, layer):
        self.layers.append(layer)

    def compile(self):
        input_shape = None
        for layer in self.layers:
            layer.build(input_shape)
            input_shape = layer.output_shape

    def fit(self, X, y, epochs=10, learning_rate=0.01, batch_size=200, penalty=None, alpha=0.1, validation_fraction=0, verbose=False):
        self.scores = {'loss': [], 'acc': []}

        if validation_fraction:
            X, y, X_val, y_val = self._validation_split(X, y, validation_fraction)
            self.scores['val_loss'] = []
            self.scores['val_acc'] = []
        else:
            X_val = None
            y_val = None

        n_samples, _ = X.shape

        if batch_size is None:
            batch_size = n_samples

        for epoch in range(epochs):
            if verbose:
                sys.stdout.write('Epoch %s/%s\n' % (epoch + 1, epochs))

            loss = 0.0
            accuracy = 0.0
            for start in range(0, len(X), batch_size):
                end = start + batch_size
                batch_slice = slice(start, end)

                output, scores = self.evaluate(X[batch_slice], y[batch_slice], True)
                loss += scores['loss']
                accuracy += scores['acc']

                if penalty is 'l2':
                    penalty_term = alpha * np.sum([np.dot(layer.weights.ravel(), layer.weights.ravel()) for layer in self.layers
                                                   if hasattr(layer, 'weights')]) / 2
                    loss += penalty_term / n_samples

                y_batch = to_categorical(y[batch_slice], output.shape[1])
                grads = cross_entropy_gradient(y_batch, output)
                self._backward(grads, learning_rate, penalty, alpha)

                if verbose:
                    sys.stdout.write('\r%*s/%s - loss: %.4f - accuracy: %.4f' % (len(str(len(X))), start + batch_size, len(X), scores['loss'], scores['acc']))
                    sys.stdout.flush()

            n_batches = math.ceil(len(X) / batch_size)
            self.scores['loss'].append(loss / n_batches)
            self.scores['acc'].append(accuracy / n_batches)

            if validation_fraction:
                _, scores = self.evaluate(X_val, y_val, training=False)
                self.scores['val_loss'].append(scores['loss'])
                self.scores['val_acc'].append(scores['acc'])

            if verbose:
                sys.stdout.write('\r%s/%s - loss: %.4f - accuracy: %.4f'
                                 % (len(X), len(X), self.scores['loss'][-1], self.scores['acc'][-1]))
                sys.stdout.flush()

            if verbose:
                sys.stdout.write('\n')

    def predict(self, X):
        return np.argmax(self._forward(X), axis=1)

    def evaluate(self, X, y, training=False):
        output = self._forward(X, training)
        _, n_classes = output.shape
        loss = np.sum(categorical_cross_entropy(to_categorical(y, n_classes), output)) / len(X)
        accuracy = np.sum(y == np.argmax(output, axis=1)) / len(X)
        return output, {'loss': loss, 'acc': accuracy}

    def _validation_split(self, X, y, fraction):
        n_val_samples = math.ceil(fraction * len(X))
        X_val = X[0:n_val_samples]
        y_val = y[0:n_val_samples]
        X = X[n_val_samples:]
        y = y[n_val_samples:]
        return X, y, X_val, y_val

    def _forward(self, X, training=False):
        output = X
        for layer in self.layers:
            output = layer.forward(output, training=training)
        return output

    def _backward(self, grads, learning_rate, penalty, alpha):
        n_samples, _ = grads.shape

        for layer in reversed(self.layers):
            weight_grads, bias_grads, grads = layer.backward(grads)

            if hasattr(layer, 'weights') and hasattr(layer, 'biases'):
                if penalty is 'l2':
                    weight_grads += alpha * layer.weights / n_samples

                layer.weights -= learning_rate * weight_grads
                layer.biases -= learning_rate * bias_grads
