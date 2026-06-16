import numpy as np

from .base import  Initializer


class He(Initializer):

    def initialize(
        self,
        input_size,
        output_size
    ):
        return (
            np.random.randn(
                input_size,
                output_size
            )
            * np.sqrt(2 / input_size)
        )