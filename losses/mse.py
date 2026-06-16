import numpy as np


class MSE:
    """The mean squared error loss function.    """

    def forward(self, y_true, y_pred):
        return np.mean(
            (y_true - y_pred) ** 2
        )

    def backward(self, y_true, y_pred):
        return (
            2 *
            (y_pred - y_true)
            / y_true.size
        )