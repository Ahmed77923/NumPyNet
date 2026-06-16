from sklearn.datasets import make_moons

from .base import Dataset


class Moons(Dataset):

    def __init__(
        self,
        n_samples=1000,
        noise=0.2,
        random_state=42
    ):
        self.n_samples = n_samples
        self.noise = noise
        self.random_state = random_state

    def load(self):

        X, y = make_moons(
            n_samples=self.n_samples,
            noise=self.noise,
            random_state=self.random_state
        )

        return X, y.reshape(-1, 1)