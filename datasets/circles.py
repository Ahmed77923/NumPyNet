from sklearn.datasets import make_circles

from .base import Dataset


class Circles(Dataset):

    def __init__(
        self,
        n_samples=1000,
        noise=0.1,
        factor=0.5,
        random_state=42
    ):
        self.n_samples = n_samples
        self.noise = noise
        self.factor = factor
        self.random_state = random_state

    def load(self):

        X, y = make_circles(
            n_samples=self.n_samples,
            noise=self.noise,
            factor=self.factor,
            random_state=self.random_state
        )

        return X, y.reshape(-1, 1)