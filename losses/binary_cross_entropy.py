import numpy as np


class BinaryCrossEntropy:
    """Binary cross-entropy loss for binary classification tasks."""

    def forward(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        return -np.mean(
            y_true * np.log(y_pred)
            + (1 - y_true) * np.log(1 - y_pred)
        )

    def backward(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        gradient = (
            -(y_true / y_pred)
            + ((1 - y_true) / (1 - y_pred))
        )
        return gradient / y_true.size
    


