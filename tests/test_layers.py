import numpy as np
from layers import Linear, initializer


class TestLinearLayer:
    def test_linear_layer(self):
        linear = Linear((3, 2), 5, initializer)
        inputs = np.ones((3, 2))
        linear.forward(inputs)
        assert linear.output.shape == (3, 5)
