import numpy as np
import math
from nn.activations import get_activation_function, apply_activation_gradients
from nn.initializers import RandomNormal, Zeros, GlorotNormal


class Layer(object):

    def __init__(self):
        self.opt_params = {}


class Dense(Layer):

    def __init__(self, units,
                 activation=None,
                 weight_initializer=RandomNormal(),
                 bias_initializer=Zeros(),
                 regularizer=None,
                 input_shape=None):
        super().__init__()
        self.units = units
        self.activation = activation
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.regularizer = regularizer
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

        self.weights = self.weight_initializer.get_values((*input_shape, self.units))
        self.biases = np.zeros(shape=self.units)
        self.output_shape = (self.units,)

    def forward(self, input, **kwargs):
        self.input = input
        output = np.matmul(self.input, self.weights) + self.biases

        if self.activation:
            activation = get_activation_function(self.activation)
            output = activation(output)

        return output

    def backward(self, grads):
        Z = np.matmul(self.input, self.weights) + self.biases

        if self.activation:
            grads = apply_activation_gradients(self.activation, Z, grads)

        n_samples, _ = grads.shape
        weight_grads = np.matmul(self.input.T, grads) / n_samples
        bias_grads = np.sum(grads, axis=0) / n_samples
        grads = np.matmul(grads, self.weights.T)
        return weight_grads, bias_grads, grads


class Conv2D(Layer):

    def __init__(self, filters,
                 kernel_size,
                 stride=(1, 1),
                 activation=None,
                 weight_initializer=GlorotNormal(),
                 bias_initializer=Zeros(),
                 input_shape=None):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation = activation
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
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

        channels, *input_size = self.input_shape
        self.weights = self.weight_initializer.get_values((self.filters, channels, *self.kernel_size))
        self.biases = self.bias_initializer.get_values((self.filters,))
        self.output_shape = (self.filters, *np.divide(input_size, self.stride).astype(int))

    def forward(self, input, **kwargs):
        input = input.reshape((-1, *self.input_shape))

        self.padded_input, self.padding = pad(input, self.kernel_size, self.stride)
        self.convs = convolve(self.padded_input, self.weights, self.stride) + self.biases.reshape(1, *self.biases.shape, 1, 1)

        if self.activation:
            activation = get_activation_function(self.activation)
            return activation(self.convs)

        return self.convs

    def backward(self, grads):
        if self.activation:
            grads = apply_activation_gradients(self.activation, self.convs, grads)

        batch_size, channels, padded_rows, padded_cols = self.padded_input.shape
        row_stride, col_stride = self.stride
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

                rows = slice(row * row_stride, row * row_stride + filter_rows)
                cols = slice(col * col_stride, col * col_stride + filter_cols)
                input_grads[:, :, :, rows, cols] += self.weights * grads[:, :, None, row, col, None, None]

        grads_reshaped = np.expand_dims(grads, 2)
        grads_reshaped = grads_reshaped.reshape((*grads_reshaped.shape, 1, 1))
        weight_grads = np.sum(input_patches * grads_reshaped, (0, 3, 4)) / batch_size
        bias_grads = np.sum(grads_reshaped, (0, 3, 4, 5, 6)).reshape(-1) / batch_size

        input_grads = np.sum(trim_padding(input_grads, self.padding), 1) / self.filters
        return weight_grads, bias_grads, input_grads


class MaxPooling2D(Layer):

    def __init__(self, pool_size=(2, 2)):
        super().__init__()
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


class Flatten(Layer):

    def __init__(self):
        super().__init__()
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


class Dropout(Layer):

    def __init__(self, keep_prob):
        super().__init__()
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


def convolve(input, filters, stride):
    n_filters, channels, filter_rows, filter_cols = filters.shape
    batch_size, channels, input_rows, input_cols = input.shape
    row_stride, col_stride = stride
    output_rows = (input_rows - filter_rows) // row_stride + 1
    output_cols = (input_cols - filter_cols) // col_stride + 1

    input_patches = np.zeros((batch_size, channels, output_cols, output_rows, filter_rows, filter_cols))

    for row in range(0, output_rows, row_stride):
        for col in range(0, output_cols, col_stride):
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