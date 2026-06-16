import numpy as np


class Sigmoid:
    """Sigmoid activation function."""
    def forward(self, x):
        """The sigmoid activation function maps any input to a value between 0 and 1."""
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, grad):
        """The gradient of the sigmoid function is sigmoid(x) * (1 - sigmoid(x)), so we multiply the incoming gradient by this value."""
        return grad * self.output * (1 - self.output)