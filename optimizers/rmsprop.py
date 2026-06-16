import numpy as np

from .base import Optimizer


class RMSProp(Optimizer):
    """RMSProp optimizer implementation."""

    def __init__(self,beta=0.9,epsilon=1e-8):
        self.beta = beta
        self.epsilon = epsilon
        self.cache = {}

    def update(self,param_name,params,grads,learning_rate):

        if param_name not in self.cache:
            self.cache[param_name] = np.zeros_like(
                params
            )

        self.cache[param_name] = (self.beta * self.cache[param_name]+ (1 - self.beta)* grads**2
        )

        return (params- learning_rate* grads/ (np.sqrt(self.cache[param_name])+ self.epsilon)
        )