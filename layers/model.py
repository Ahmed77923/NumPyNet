import pickle

import numpy as np

from datasets.moons import Moons
from datasets.circles import Circles

from core.DenseLayer import DenseLayer

from Initializer.He import He
from Initializer.xavier import Xavier

from losses.mse import MSE

from optimizers.sgd import SGD
from optimizers.adam import Adam
from optimizers.rmsprop import RMSProp
from optimizers.momentum import Momentum

from regularization.l2 import L2
from regularization.l1 import L1

from visualization.losses_plot import plot_loss_curve
from visualization.visualize_network import visualize_network

from experiments.compare import compare

class Model:

    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def forward(self, X):

        output = X

        for layer in self.layers:
            output = layer.forward(output)

        return output

    def backward(self, grad, learning_rate, optimizer):
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate, optimizer)

    def save(self, path):

        with open(path, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(path):

        with open(path, "rb") as file:
            return pickle.load(file)

    def summary(self):

        print("\nModel Summary")
        print("-" * 50)

        total_params = 0

        print(
            f"{'Layer':<15}"
            f"{'Input':<10}"
            f"{'Output':<10}"
            f"{'Params':<10}"
        )

        print("-" * 50)

        for layer in self.layers:

            params = (
                layer.weights.size
                +
                layer.bias.size
            )

            total_params += params

            print(
                f"{layer.__class__.__name__:<15}"
                f"{layer.weights.shape[0]:<10}"
                f"{layer.weights.shape[1]:<10}"
                f"{params:<10}"
            )

        print("-" * 50)

        print(
            f"Total Parameters: {total_params}"
        )







