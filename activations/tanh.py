import numpy as np


class Tanh:
    """Tanh activation function."""

    def forward(self, x):
        """The tanh activation function maps any input to a value between -1 and 1."""
        self.output = np.tanh(x)
        return self.output

    def backward(self, grad):
        """The gradient of the tanh function is 1 - tanh(x)^2, so we multiply the incoming gradient by this value."""
        return grad * (1 - self.output ** 2)