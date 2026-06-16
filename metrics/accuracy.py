import numpy as np

from .base import Metric


class Accuracy(Metric):

    def __init__(
        self,
        threshold=0.5
    ):
        self.threshold = threshold

    def compute(
        self,
        y_true,
        y_pred
    ):

        y_pred = (
            y_pred > self.threshold
        ).astype(int)

        return np.mean(
            y_true == y_pred
        )