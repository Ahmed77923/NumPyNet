from abc import ABC, abstractmethod


class Initializer(ABC):

    @abstractmethod
    def initialize(
        self,
        input_size,
        output_size
    ):
        pass