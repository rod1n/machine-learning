import sys
import numpy as np
import math


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

                y_batch = one_hot(y[batch_slice], output.shape[1])
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
        loss = np.sum(cross_entropy(one_hot(y, n_classes), output)) / len(X)
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


class Dense(object):

    def __init__(self, units, activation=None, input_shape=None):
        self.units = units
        self.activation = activation
        self.input_shape = input_shape
        self.input = None
        self.weights = None
        self.biases = None
        self.output_shape = None

    def build(self, input_shape):
        if input_shape is None:
            input_shape = self.input_shape

        if input_shape is None:
            raise ValueError('Specify input_shape parameter')

        self.weights = np.random.normal(0.0, 1.0, size=(*input_shape, self.units))
        self.biases = np.zeros(shape=self.units)
        self.output_shape = (self.units,)

    def forward(self, input, **kwargs):
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

    def __init__(self, filters, kernel_size, activation=None, input_shape=None):
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.input_shape = input_shape
        self.weights = None
        self.biases = None
        self.output_shape = None
        self.padded_input = None
        self.padding = None
        self.convs = None

    def build(self, input_shape):
        if self.input_shape is None:
            if input_shape is None:
                raise ValueError('Specify input_shape parameter')
            self.input_shape = input_shape

        channels, rows, cols = self.input_shape
        self.weights = np.random.normal(0.0, 1.0, size=(self.filters, channels, *self.kernel_size))
        self.biases = np.zeros(shape=(self.filters))
        self.output_shape = (self.filters, rows, cols)

    def forward(self, input, **kwargs):
        input = input.reshape((-1, *self.input_shape))

        self.padded_input, self.padding = pad(input, self.kernel_size)
        self.convs = convolve(self.padded_input, self.weights) + self.biases.reshape(1, *self.biases.shape, 1, 1)

        if self.activation:
            return apply_activation(self.activation, self.convs)
        else:
            return self.convs

    def backward(self, grads):
        if self.activation:
            grads = get_activation_gradient(self.activation, self.convs, grads)

        batch_size, channels, padded_rows, padded_cols = self.padded_input.shape
        _, output_rows, output_cols = self.output_shape
        filter_rows, filter_cols = self.kernel_size

        input_patches = np.zeros((batch_size, 1, channels, output_rows, output_cols, filter_rows, filter_cols))
        input_grads = np.zeros((batch_size, self.filters, channels, padded_rows, padded_cols))

        for row in range(output_rows):
            for col in range(output_cols):
                rows = slice(row, row + filter_rows)
                cols = slice(col, col + filter_cols)

                patch = self.padded_input[:, :, rows, cols]
                input_patches[:, 0, :, row, col] = patch
                input_grads[:, :, :, rows, cols] += self.weights * grads[:, :, None, row, col, None, None]

        grads_reshaped = np.expand_dims(grads, 2)
        grads_reshaped = grads_reshaped.reshape((*grads_reshaped.shape, 1, 1))
        weight_grads = np.sum(input_patches * grads_reshaped, (0, 3, 4)) / batch_size
        bias_grads = np.sum(grads_reshaped, (0, 3, 4, 5, 6)).reshape(-1) / batch_size

        input_grads = np.sum(trim_padding(input_grads, self.padding), 1) / self.filters
        return weight_grads, bias_grads, input_grads


class MaxPooling2D(object):

    def __init__(self, pool_size=(2, 2)):
        self.pool_size = pool_size
        self.pool_mask = None
        self.input_shape = None
        self.output_shape = None
        self.padding = None

    def build(self, input_shape):
        if input_shape is None:
            raise ValueError('Specify input_shape parameter')

        self.input_shape = input_shape
        self.output_shape = np.ceil(np.divide(input_shape, (1, *self.pool_size))).astype(int)

    def forward(self, input, **kwargs):
        padded_input, self.padding = pad(input, self.pool_size, stride=self.pool_size)

        batch_size, *_ = input.shape
        pool_rows, pool_cols = self.pool_size
        n_filters, output_rows, output_cols = self.output_shape

        self.pool_mask = np.zeros((batch_size, *self.output_shape, pool_rows, pool_cols))
        output = np.zeros((batch_size, *self.output_shape))

        for row in range(output_rows):
            for col in range(output_cols):
                rows = slice(row * pool_rows, row * pool_rows + pool_rows)
                cols = slice(col * pool_cols, col * pool_cols + pool_cols)

                patches = padded_input[:, :, rows, cols]
                output[:, :, row, col] = np.max(patches, axis=(2, 3))
                self.pool_mask[:, :, row, col] = np.max(patches, axis=(2, 3), keepdims=True) == patches
        return output

    def backward(self, grads):
        grads = self.pool_mask * grads.reshape((*grads.shape, 1, 1))

        batch_size, *_ = grads.shape
        padded_shape = np.multiply(self.output_shape, (1, *self.pool_size))
        _, output_rows, output_cols = self.output_shape
        pool_rows, pool_cols = self.pool_size

        input_grads = np.zeros(shape=(batch_size, *padded_shape))
        for row in range(output_rows):
            for col in range(output_cols):
                rows = slice(row * pool_rows, row * pool_rows + pool_rows)
                cols = slice(col * pool_cols, col * pool_cols + pool_cols)
                input_grads[:,:, rows, cols] = grads[:,:, row, col]

        (left, right), (top, bottom) = self.padding
        _, input_rows, input_cols = self.input_shape
        input_grads = input_grads[:, :, top:top + input_rows, left:left + input_cols]
        return None, None, input_grads


class Flatten(object):

    def __init__(self):
        self.input_shape = None
        self.output_shape = None

    def build(self, input_shape):
        if input_shape is None:
            raise ValueError('Specify input_shape parameter')

        self.input_shape = input_shape
        self.output_shape = (np.prod(input_shape),)

    def forward(self, input, **kwargs):
        return input.reshape((-1, *self.output_shape))

    def backward(self, grads):
        grads = grads.reshape((-1, *self.input_shape))
        return None, None, grads


class Dropout(object):

    def __init__(self, keep_prob):
        self.keep_prob = keep_prob
        self.mask = None
        self.output_shape = None

    def build(self, input_shape):
        if input_shape is None:
            raise ValueError('Specify input_shape parameter')
        self.output_shape = input_shape

    def forward(self, input, training=False):
        _, *sample_shape = input.shape
        if training:
            self.mask = np.random.binomial(1, self.keep_prob, size=sample_shape) / self.keep_prob
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
    n_filters, channels, filter_rows, filter_cols = filters.shape
    batch_size, channels, input_rows, input_cols = input.shape
    output_rows = input_rows - filter_rows + 1
    output_cols = input_cols - filter_cols + 1

    input_patches = np.zeros((batch_size, channels, output_cols, output_rows, filter_rows, filter_cols))

    for row in range(output_rows):
        for col in range(output_cols):
            rows = slice(row, row + filter_rows)
            cols = slice(col, col + filter_cols)
            patch = input[:, :, rows, cols]
            input_patches[:, :, row, col, :, :] = patch

    return np.einsum('bcijnm,fcnm->bfcij', input_patches, filters, optimize=True).sum(axis=2)


def pad(input, filter_shape, stride=(1, 1)):
    *_, input_height, input_widths = input.shape
    filter_height, filter_width = filter_shape
    stride_height, stride_widths = stride

    if input_height % stride_height == 0:
        height_padding = np.max(filter_height - stride_height, 0)
    else:
        height_padding = np.max(filter_height - input_height % stride_widths, 0)

    if input_widths % stride_widths == 0:
        width_padding = np.max(filter_width - stride_widths, 0)
    else:
        width_padding = np.max(filter_width - input_widths % stride_widths, 0)

    left = math.ceil(width_padding / 2)
    right = math.floor(width_padding / 2)
    top = math.ceil(height_padding / 2)
    bottom = math.floor(height_padding / 2)

    zero_paddings = (input.ndim - 2) * [(0, 0)]
    input_padding = ((left, right), (top, bottom))
    output = np.pad(input, (*zero_paddings, *input_padding), mode='constant')
    return output, input_padding


def trim_padding(input, padding):
    (left, right), (top, bottom) = padding

    if left is 0:
        left = None
    if right is 0:
        right = None
    else:
        right = -right
    if top is 0:
        top = None
    if bottom is 0:
        bottom = None
    else:
        bottom = -bottom
    return input[..., top:bottom, left:right]