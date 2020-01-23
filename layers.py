import numpy as np


class Layer:
    def __init__(self):
        self.params = {}
        self.grads = {}

    def forward(self, inputs):
        raise NotImplementedError

    def backward(self, grad):
        raise NotImplementedError


def initializer(n_in, n_out, kind='xavier'):
    params = {}
    params['W'] = np.random.randn(n_in, n_out) / (np.sqrt(n_in))
    params['b'] = np.zeros(n_out)
    return params


class Linear(Layer):
    def __init__(self, input_shape, n_out, initializer):
        super().__init__()
        self.params = initializer(input_shape[1], n_out)

    def forward(self, inputs):
        self.inputs = inputs
        print(self.inputs)
        self.output = self.inputs @ self.params['W'] + self.params['b']
        return self.output

    def backward(self, upstream_grad):
        """
        Y = XW + b
        dY/dW = X.T * dL/dY
        dY/dX = dL/dY * W.T
        """
        self.grads['W'] = self.inputs.T @ upstream_grad
        self.grads['b'] = np.sum(upstream_grad, axis=0)
        self.grads['input'] = self.params['W'].T @ upstream_grad

    def update_params(self, learning_rate=0.1):
        self.params['W'] = self.params['W'] - learning_rate * self.dW
        self.params['b'] = self.params['b'] - learning_rate * self.db


class Sigmoid(Layer):

    def __init__(self, shape):
        super().__init__()
        self.output = np.zeros(shape)

    def forward(self, inputs):
        self.inputs = inputs
        return 1 / (1 + np.exp(-inputs))

    def backward(self, upstream_grad):
        return self.inputs * (1 - self.inputs)
