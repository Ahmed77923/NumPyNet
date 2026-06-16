import numpy as np

from activations.sigmoid import Sigmoid
from activations.linear import Linear
from activations.relu import Relu
from activations.tanh import Tanh
from optimizers.base import Optimizer

class DenseLayer:

    def __init__(
        self,
        input_size,
        output_size,
        initializer,
        activation="sigmoid", 
        regularizer = None
    ):
        self.input_size = input_size
        self.output_size = output_size  
        
        self.regularizer = regularizer

        self.weights = initializer.initialize(
            input_size,
            output_size
        )

        self.bias = np.zeros(
            (1, output_size)
        )

        if activation == "sigmoid":
            self.activation = Sigmoid()

        elif activation == "linear":
            self.activation = Linear()

        elif activation == "relu":
            self.activation = Relu()

        elif activation == "tanh":
            self.activation = Tanh()

        else:
            raise ValueError(
                f"Unsupported activation: {activation}"
            )

    def forward(self, inputs):

        self.inputs = inputs

        self.z = inputs @ self.weights + self.bias

        self.output = self.activation.forward(
            self.z
        )

        return self.output
    def backward(self, output_error, learning_rate, optimizer):
        activation_error = self.activation.backward(
            output_error
        ) # -> dZ = dA * sigmoid'(Z)
        # the formeula in forward ->    X  ->  Z = X @ W + b   ->   Activation  ->  loss
        # the formeula in backward ->   dX  <-  d_Z = X @ W + b   <-    d_Activation  <-  loss
        weights_error = self.inputs.T @ activation_error    # dW = Xᵀ · dZ
        
        if self.regularizer is not None:
            weights_error += self.regularizer.gradient(self.weights) # Add L2 regularization gradient

        inputs_error = activation_error @ self.weights.T     # dX = dZ · Wᵀ




        self.weights = optimizer.update(
            f"{id(self)}_W",
            self.weights,
            weights_error,
            learning_rate
        )

        self.bias = optimizer.update(
            f"{id(self)}_b",
            self.bias,
            np.sum(
                activation_error,
                axis=0,
                keepdims=True
            ),
            learning_rate,
        )

        return inputs_error