import numpy as np

from .base import Initializer


class Zero(Initializer):

    def initialize(
        self,
        input_size,
        output_size
    ):
        return np.zeros(
            (input_size, output_size)
        )