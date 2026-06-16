import numpy as np

from .base import Metric


class Precision(Metric):

    def __init__(self,threshold=0.5):
        self.threshold = threshold

    def compute(self,y_true,y_pred):

        y_pred = (y_pred > self.threshold).astype(int)

        tp = np.sum((y_true == 1)&(y_pred == 1))

        fp = np.sum((y_true == 0)&(y_pred == 1))

        return tp / (tp + fp + 1e-15)