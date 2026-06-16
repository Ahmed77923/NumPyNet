import numpy as np

from .base import Initializer


class Xavier(Initializer):

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
            * np.sqrt(1 / input_size)
        )