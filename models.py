class Sequential:

    def __init__(self, steps, cost_fn):
        self.steps = steps
        self.cost_fn = cost_fn

    def train(self, X, y, learning_rate, epochs):
        step = self.steps[0]
        step.forward(X)
        for step, idx in enumerate(self.steps[1:]):
            step.forward(self.step[idx].output)
