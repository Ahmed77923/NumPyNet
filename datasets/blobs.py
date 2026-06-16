from sklearn.datasets import make_blobs

from .base import Dataset


class Blobs(Dataset):

    def __init__(
        self,
        n_samples=1000,
        centers=2,
        cluster_std=1.0,
        random_state=42
    ):
        self.n_samples = n_samples
        self.centers = centers
        self.cluster_std = cluster_std
        self.random_state = random_state

    def load(self):

        X, y = make_blobs(
            n_samples=self.n_samples,
            centers=self.centers,
            cluster_std=self.cluster_std,
            random_state=self.random_state
        )

        return X, y.reshape(-1, 1)