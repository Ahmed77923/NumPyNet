import numpy as np

from .base import Optimizer


class Adam(Optimizer):
    """Adam optimizer implementation."""

    def __init__(self,beta1=0.9,beta2=0.999,epsilon=1e-8
    ):

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = {}
        self.v = {}

        self.t = 0

    def update(self,param_name,params,grads,learning_rate
    ):

        self.t += 1

        if param_name not in self.m:

            self.m[param_name] = np.zeros_like(
                params
            )

            self.v[param_name] = np.zeros_like(
                params
            )

        self.m[param_name] = (self.beta1* self.m[param_name]+ (1 - self.beta1)* grads
        )

        self.v[param_name] = (self.beta2* self.v[param_name]+ (1 - self.beta2)* grads**2
        )

        m_hat = (
            self.m[param_name]/ (1- self.beta1**self.t)
        )

        v_hat = (
            self.v[param_name]/ (1- self.beta2**self.t)
        )

        return (params- learning_rate* m_hat/ (np.sqrt(v_hat)+ self.epsilon
            )
        )