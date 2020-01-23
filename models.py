import numpy as np

class initializer:
    def __init__(self, n_in, n_out, kind='xavier'):
        self.n_in = n_in
        self.n_out = n_out
        self.kinc = kind
    
    def get_params(self):
        params = {}
        params['W'] = np.random.randn(self.n_out, self.n_in) / (np.sqrt(self.n_in))
        params['b'] = np.zeros((self.n_out, 1))
        return params
    
    __call__ = get_params


class LinearLayer:
    def __init__(self, input_shape, n_out, initializer):
        self.params = initializer(input_shape, n_out)()
    
    def forward(self, output_prev):
        self.output_prev = output_prev
        self.output = np.dot(self.params('W'), self.output_prev) + self.params['b']
    
    def backward(self, upstream_grad):
        self.dW = np.dot(upstream_grad, self.output_prev.T)
        self.db = np.sum(upstream_grad, axis=1, keepdims=True)
        self.d_output_prev = np.dot(self.params['W'].T, upstream_grad)
    
    def update_params(self, learning_rate=0.1):
        self.params['W'] = self.params['W'] - learning_rate * self.dW
        self.params['b'] = self.params['b'] - learning_rate * self.db

class SigmoidLayer:

    def __init__(self, shape):
        self.output = np.zeros(shape)
    
    def forward(output_prev):
        self.output = 1 / (1 + np.exp(-output_prev))
    
    def backward(self, upstream_grad):
        self.d = upstream_grad * self.A*(1-self.A)

class Sequential:

    def __init__(self, steps, cost_fn):
        self.steps = steps
        self.cost_fn = cost_fn
    
    def train(self, X, y, learning_rate, epochs):
        step = self.steps[0]
        step.forward(X)
        for step, idx in enumerate(self.steps[1:]):
            step.forward(self.step[idx].output)
        
