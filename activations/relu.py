import numpy as np


class Relu:
    """ReLU activation function."""
    def forward(self, x):
        """The ReLU activation function returns the input if it is positive, and 0 otherwise."""
        self.input = x
        return np.maximum(0, x)

    def backward(self, grad):
        """The gradient of the ReLU function is 1 if the input is positive, and 0 otherwise."""
        return grad * (self.input > 0)