from .base import Regularizer

# Ridge regression (L2 regularization)
class L2(Regularizer):

    def __init__(self, lamda=0.01):
        self.lamda = lamda

    def penalty(self, W):
        return 0.5 * self.lamda * (W ** 2).sum()

    def gradient(self, W):
        return self.lamda * W