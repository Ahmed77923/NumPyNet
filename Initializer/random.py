import numpy as np

from .base import Initializer


class Random(Initializer):

    def initialize(
        self,
        input_size,
        output_size
    ):
        return (
            np.random.randn(
                input_size,
                output_size
            ) * 0.01
        )