class Linear:
    """Linear activation function."""

    def forward(self, x):
        """The linear activation function simply returns the input as output."""
        return x

    def backward(self, grad):
        """The gradient of the linear activation function is 1, so we simply return the incoming gradient."""
        return grad