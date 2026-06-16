import numpy as np

# Lasso regression (L1 regularization) 
from .base import Regularizer
class L1(Regularizer):
    """L1 regularization class that implements the L1 penalty and its gradient."""
    def __init__(self, lamda=0.01):
        self.lamda = lamda

    def penalty(self, weights):
        return self.lamda * np.sum(np.abs(weights))

    def gradient(self, weights):
        return self.lamda * np.sign(weights)