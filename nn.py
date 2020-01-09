import sys
import numpy as np
import math


class Model(object):

    def __init__(self):
        self.layers = []
        self.scores = None

    def add(self, layer):
        self.layers.append(layer)

    def fit(self, X, y, epochs=10, learning_rate=0.01, batch_size=200, penalty=None, alpha=0.1, validation_fraction=0, verbose=False):
        self.scores = {'loss': [], 'acc': []}

        if validation_fraction:
            X, y, X_val, y_val = self._validation_split(X, y, validation_fraction)
            self.scores['val_loss'] = []
            self.scores['val_acc'] = []
        else:
            X_val = None
            y_val = None

        n_samples, n_features = X.shape
        self._initialize(n_features)

        if batch_size is None:
            batch_size = n_samples

        for epoch in range(epochs):
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
                                                   if layer.has_trainable_variables()]) / 2
                    loss += penalty_term / n_samples

                y_batch = one_hot(y[batch_slice], output.shape[1])
                grads = cross_entropy_gradient(y_batch, output)
                self._backward(grads, learning_rate, penalty, alpha)

            n_batches = math.ceil(len(X) / batch_size)
            self.scores['loss'].append(loss / n_batches)
            self.scores['acc'].append(accuracy / n_batches)

            if validation_fraction:
                _, scores = self.evaluate(X_val, y_val, training=False)
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

    def evaluate(self, X, y, training=False):
        output = self._forward(X, training)
        _, n_classes = output.shape
        loss = np.sum(cross_entropy(one_hot(y, n_classes), output)) / len(X)
        accuracy = np.sum(y == np.argmax(output, axis=1)) / len(X)
        return output, {'loss': loss, 'acc': accuracy}

    def _initialize(self, input_shape):
        for layer in self.layers:
            if layer.has_trainable_variables():
                if isinstance(layer, Dense):
                    layer.weights = np.random.normal(0.0, 1.0, size=(input_shape, layer.units))
                    layer.biases = np.zeros(shape=layer.units)
                    input_shape = layer.units
                elif isinstance(layer, Conv2D):
                    layer.weights = np.random.normal(0.0, 1.0, size=(layer.filters, *layer.kernel_size))
                    layer.biases = np.zeros(shape=layer.filters)
                    input_shape = (layer.filters, *layer.input_shape)
            else:
                if isinstance(layer, Flatten):
                    input_shape = np.prod(input_shape)

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

            if penalty is 'l2':
                if layer.has_trainable_variables():
                    weight_grads += alpha * layer.weights / n_samples

            if layer.has_trainable_variables():
                layer.weights -= learning_rate * weight_grads
                layer.biases -= learning_rate * bias_grads


class Dense(object):

    def __init__(self, units, activation=None):
        self.units = units
        self.activation = activation
        self.input = None
        self.weights = None
        self.biases = None

    def has_trainable_variables(self):
        return True

    def forward(self, input, training=False):
        self.input = input
        Z = np.matmul(self.input, self.weights) + self.biases

        if self.activation:
            return apply_activation(self.activation, Z)
        else:
            return Z

    def backward(self, grads):
        Z = np.matmul(self.input, self.weights) + self.biases

        if self.activation:
            grads = get_activation_gradient(self.activation, Z, grads)

        n_samples, _ = grads.shape
        weight_grads = np.matmul(self.input.T, grads) / n_samples
        bias_grads = np.sum(grads, axis=0) / n_samples
        grads = np.matmul(grads, self.weights.T)
        return weight_grads, bias_grads, grads


class Conv2D(object):

    def __init__(self, filters, kernel_size, activation=None, **kwargs):
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.weights = None
        self.biases = None
        self.padded_input = None
        self.padding = None
        self.convs = None

        if 'input_shape' in kwargs:
            self.input_shape = kwargs['input_shape']

    def has_trainable_variables(self):
        return True

    def forward(self, input, **kwargs):
        input = input.reshape((-1, *self.input_shape))

        self.padded_input, self.padding = pad(input, self.kernel_size)
        self.convs = convolve(self.padded_input, self.weights)

        if self.activation:
            return apply_activation(self.activation, self.convs)
        else:
            return self.convs

    def backward(self, grads):
        if self.activation:
            grads = get_activation_gradient(self.activation, self.convs, grads)

        n_samples, padded_rows, padded_cols = self.padded_input.shape
        output_rows, output_cols = self.input_shape
        filter_rows, filter_cols = self.kernel_size

        input_patches = np.zeros((n_samples, 1, output_rows, output_cols, filter_rows, filter_cols))
        input_grads = np.zeros((self.filters, padded_rows, padded_cols))

        for row in range(output_rows):
            for col in range(output_cols):
                rows = slice(row, row + filter_rows)
                cols = slice(col, col + filter_cols)

                patch = self.padded_input[:, rows, cols]
                input_patches[:, 0, row, col] = patch
                input_grads[:, rows, cols] += self.weights

        grads = grads.reshape((*grads.shape, 1, 1))
        weight_grads = np.sum(input_patches * grads, (0, 2, 3)) / n_samples
        bias_grads = np.sum(grads, (0, 2, 3, 4, 5)) / n_samples

        left, top, *_ = self.padding
        input_grads = input_grads[top:top + output_rows, left:left + output_cols]

        return weight_grads, bias_grads, input_grads


class Flatten(object):

    def __init__(self):
        self.sample_shape = None

    def has_trainable_variables(self):
        return False

    def forward(self, input, **kwargs):
        n_samples, *self.sample_shape = input.shape
        return input.reshape((n_samples, -1))

    def backward(self, grads):
        return None, None, grads.reshape((-1, *self.sample_shape))


class Dropout(object):

    def __init__(self, keep_prob):
        self.keep_prob = keep_prob
        self.mask = None

    def has_trainable_variables(self):
        return False

    def forward(self, input, training=False):
        _, units = input.shape
        if training:
            self.mask = np.random.binomial(1, self.keep_prob, size=units) / self.keep_prob
            return input * self.mask
        else:
            return input

    def backward(self, grads):
        return None, None, grads * self.mask


def apply_activation(name, values):
    if name is 'softmax':
        return softmax(values)
    elif name is 'sigmoid':
        return sigmoid(values)
    else:
        raise ValueError("Unknown activation function '%s'" % name)


def get_activation_gradient(name, values, grads):
    if name is 'softmax':
        jacobian = softmax_gradient(values)
        return np.einsum('ijk,ik->ij', jacobian, grads)
    elif name is 'sigmoid':
        return grads * sigmoid_gradient(values)
    else:
        raise ValueError('Activation function "%s" is not supported' % name)


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


def convolve(input, filters):
    n_filters, filter_rows, filter_cols = filters.shape
    batch_size, input_rows, input_cols = input.shape
    output_rows = input_rows - filter_rows + 1
    output_cols = input_cols - filter_cols + 1

    input_patches = np.zeros((batch_size, output_cols, output_rows, filter_rows, filter_cols))

    for row in range(output_rows):
        for col in range(output_cols):
            rows = slice(row, row + filter_rows)
            cols = slice(col, col + filter_cols)
            patch = input[:, rows, cols]
            input_patches[:, row, col, :, :] = patch

    return np.einsum('bijnm,fnm->bfij', input_patches, filters)


def pad(input, filter_shape):
    h_pad, v_pad = np.subtract(filter_shape, 1) / 2
    left_pad = np.ceil(h_pad).astype(int)
    right_pad = np.floor(h_pad).astype(int)
    top_pad = np.ceil(v_pad).astype(int)
    bottom_pad = np.floor(v_pad).astype(int)
    result = np.pad(input, ((0, 0), (left_pad, right_pad), (top_pad, bottom_pad)), mode='constant')
    padding = (left_pad, top_pad, right_pad, bottom_pad)
    return result, padding
