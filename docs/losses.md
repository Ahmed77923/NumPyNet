# Losses

The loss functions in NumPyNet define how model predictions are compared to labels during training.

## Implemented Losses

### MSE

Mean squared error is used for regression-style training or simple output fitting.

### Binary Cross Entropy

Binary cross entropy is intended for binary classification tasks where outputs represent probabilities.

## How Losses Are Used

The trainer expects a loss object with two operations:

- `forward(y_true, y_pred)` computes the scalar loss value
- `backward(y_true, y_pred)` returns the gradient used for backpropagation

## Choosing a Loss

- Use `MSE` for continuous targets or when the example is built around squared error
- Use binary cross entropy when training a classifier with sigmoid outputs

## Practical Note

The example in `core/trainer.py` currently uses `MSE()` for the moons dataset, which is fine for a compact demo, but binary cross entropy is usually the more natural choice for classification.
