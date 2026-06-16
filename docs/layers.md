# Layers

NumPyNet uses a small set of layer abstractions for forward and backward propagation.

## DenseLayer

`core/DenseLayer.py` provides the main trainable layer used throughout the project.

### Responsibilities

- Stores weights and bias
- Applies an activation function
- Runs forward propagation
- Computes gradients during backpropagation
- Updates parameters through the selected optimizer
- Applies optional regularization gradients

### Constructor

```python
DenseLayer(
    input_size,
    output_size,
    initializer,
    activation="sigmoid",
    regularizer=None,
)
```

### Supported Activations

- `sigmoid`
- `linear`
- `relu`
- `tanh`

## Model Container

`layers/model.py` defines the `Model` class used for assembling a full network.

### Methods

- `add(layer)` appends a layer to the network
- `forward(X)` runs inference through all layers
- `backward(grad, learning_rate, optimizer)` propagates gradients backward
- `save(path)` serializes the model with pickle
- `load(path)` restores a saved model
- `summary()` prints a parameter summary for each layer

## Typical Stack

A common binary-classification model looks like this:

```python
model = Model()
model.add(DenseLayer(2, 16, He(), "relu"))
model.add(DenseLayer(16, 8, He(), "relu"))
model.add(DenseLayer(8, 1, Xavier(), "sigmoid"))
```

## Notes

- The codebase currently focuses on feed-forward dense networks.
- Regularization is applied per layer when the layer receives a `regularizer` instance.
- The `summary()` method is useful for checking the parameter count before training.
