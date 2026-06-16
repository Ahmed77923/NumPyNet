class Regularizer:

    def penalty(self, weights):
        raise NotImplementedError

    def gradient(self, weights):
        raise NotImplementedError