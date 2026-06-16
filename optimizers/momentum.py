import numpy as np

from .base import Optimizer


class Momentum(Optimizer):
    """Momentum optimizer implementation."""

    def __init__(self,beta=0.9):
        self.beta = beta
        self.velocity = {}

    def update(self,param_name,params,grads,learning_rate
    ):

        if param_name not in self.velocity:
            self.velocity[param_name] = np.zeros_like(params)

        self.velocity[param_name] = (self.beta* self.velocity[param_name]+ (1 - self.beta)* grads
        )

        return (params- learning_rate* self.velocity[param_name]
        )