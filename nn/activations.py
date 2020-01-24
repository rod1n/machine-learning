import numpy as np


def get_activation_function(activation):
    if activation is 'softmax':
        return softmax
    elif activation is 'sigmoid':
        return sigmoid
    else:
        raise ValueError("The activation '%s' is not supported" % activation)


def apply_activation_gradients(activation, values, grads):
    if activation is 'softmax':
        jacobian = softmax_gradient(values)
        return np.einsum('ijk,ik->ij', jacobian, grads)
    elif activation is 'sigmoid':
        return sigmoid_gradient(values) * grads
    else:
        raise ValueError("The activation '%s' is not supported" % activation)


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def softmax_gradient(x):
    s = softmax(x)
    outer = np.einsum('ij,ik->ijk', s, s)
    diagonal = np.einsum('ij,jk->ijk', s, np.eye(s.shape[1]))
    jacobian = diagonal - outer
    return jacobian


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_gradient(x):
    s = sigmoid(x)
    return s * (1 - s)
