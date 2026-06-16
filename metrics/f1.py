from .base import Metric
from .precision import Precision
from .recall import Recall


class F1_score(Metric):

    def __init__(
        self,
        threshold=0.5
    ):
        self.threshold = threshold

        self.precision = Precision(threshold)

        self.recall = Recall(threshold)

    def compute(self,y_true,y_pred):

        p = self.precision(y_true,y_pred)

        r = self.recall(y_true,y_pred)

        return (2 * p * r/(p + r + 1e-15)
        )