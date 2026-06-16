# NumPyNet

NumPyNet is a lightweight neural network library built with NumPy. It provides the basic building blocks needed to experiment with feed-forward models, binary classification, custom optimizers, weight initialization, regularization, metrics, and visualizations.

The codebase is organized as a small deep-learning toolkit rather than a single training script. You can assemble models layer by layer, choose an initializer and optimizer, train on synthetic datasets, evaluate performance, and visualize results.

## What This Project Includes

- Dense layers with configurable activation functions
- Weight initializers such as He, Xavier, Random, and Zero
- Loss functions for regression and classification
- Optimizers including SGD, Momentum, RMSProp, and Adam
- Regularization with L1 and L2 penalties
- Metrics for accuracy, precision, recall, and F1 score
- Synthetic datasets for moons, circles, and blobs
- Visualization helpers for decision boundaries, loss curves, and network diagrams
- Model saving and loading with pickle

## Project Structure

```text
NumPyNet/
├── activations/
├── core/
├── datasets/
├── experiments/
├── Initializer/
├── layers/
├── losses/
├── metrics/
├── optimizers/
├── regularization/
└── visualization/
```

## Architecture

```text
NumPyNet
│
├── core
│   ├── trainer.py
│   ├── DenseLayer.py
│   └── model.py
│
├── activations
│   ├── sigmoid.py
│   ├── relu.py
│   ├── tanh.py
│   ├── linear.py
│   └── __init__.py
│
├── Initializer
│   ├── He.py
│   ├── xavier.py
│   ├── random.py
│   ├── zero.py
│   └── __init__.py
│
├── losses
│   ├── mse.py
│   ├── binary_cross_entropy.py
│   └── __init__.py
│
├── optimizers
│   ├── sgd.py
│   ├── momentum.py
│   ├── rmsprop.py
│   ├── adam.py
│   └── __init__.py
│
├── regularization
│   ├── l1.py
│   ├── l2.py
│   └── base.py
│
├── metrics
│   ├── accuracy.py
│   ├── precision.py
│   ├── recall.py
│   ├── f1.py
│   └── base.py
│
├── datasets
│   ├── moons.py
│   ├── circles.py
│   ├── blobs.py
│   └── base.py
│
├── visualization
│   ├── visualize_network.py
│   ├── losses_plot.py
│   └── decision_boundary.py
│
├── experiments
│   └── compare.py
│
└── docs
    ├── getting_started.md
    ├── layers.md
    ├── losses.md
    ├── optimizers.md
    └── examples.md
```

The architecture is intentionally small and modular: datasets produce `X, y`, `DenseLayer` blocks transform the data, the `Model` container chains layers together, `Trainer` runs optimization and metrics, and visualization helpers show the result or save the trained network.

### Main Packages

- `core/`: training logic and dense-layer implementation used by the examples
- `layers/`: reusable model container with `forward`, `backward`, `save`, `load`, and `summary`
- `datasets/`: synthetic dataset loaders built on top of `scikit-learn`
- `Initializer/`: weight initialization strategies
- `losses/`: mean squared error and binary cross entropy
- `optimizers/`: optimization algorithms for gradient descent training
- `regularization/`: L1 and L2 regularization
- `metrics/`: evaluation metrics used during training and validation
- `visualization/`: plotting helpers for model behavior and training curves
- `experiments/`: comparison helpers for optimizer experiments

## Requirements

The project uses:

- Python 3.10+ recommended
- NumPy
- scikit-learn
- Matplotlib

Install the dependencies with:

```bash
pip install numpy scikit-learn matplotlib
```

## Getting Started

The usual workflow is:

1. Load a dataset
2. Build a model by adding layers
3. Choose a loss function and optimizer
4. Train the model
5. Evaluate metrics
6. Visualize the decision boundary or loss curve
7. Save the model if needed

### Quick Example

```python
from datasets.moons import Moons
from core.DenseLayer import DenseLayer
from Initializer.He import He
from Initializer.xavier import Xavier
from losses.mse import MSE
from optimizers.adam import Adam
from layers.model import Model
from core.trainer import Trainer

X, y = Moons(n_samples=1000, noise=0.1).load()

model = Model()
model.add(DenseLayer(2, 16, He(), "relu"))
model.add(DenseLayer(16, 8, He(), "relu"))
model.add(DenseLayer(8, 1, Xavier(), "sigmoid"))

trainer = Trainer(
    model=model,
    loss_fn=MSE(),
    optimizer=Adam(),
    learning_rate=0.01,
)

history = trainer.fit(X, y, epochs=5000)
metrics = trainer.evaluate(X, y)

print(metrics)
model.save("moon.pkl")
loaded_model = Model.load("moon.pkl")
```

## Module Summary

### Core

`core/DenseLayer.py` defines the dense layer used in the training examples. It supports forward and backward propagation, activation selection, optional regularization, and optimizer-driven parameter updates.

`core/trainer.py` wraps the training loop, computes loss and metrics, and provides a simple `train(...)` helper for running experiments.

### Activations

The activation package contains the non-linear functions used by dense layers:

- `linear`
- `relu`
- `sigmoid`
- `tanh`

These activations let you switch between regression-style outputs and binary classification outputs.

### Initializers

The initializer package contains strategies for building stable weight matrices:

- `Zero`
- `Random`
- `Xavier`
- `He`

He initialization is useful for ReLU networks, while Xavier initialization works well for balanced activations.

### Losses

The current loss functions are:

- `MSE` for regression-style training
- `Binary Cross Entropy` for classification tasks

The training loop uses the selected loss object for both forward loss computation and gradient calculation.

### Optimizers

The optimizer package includes:

- `SGD`
- `Momentum`
- `RMSProp`
- `Adam`

All optimizers follow the same update pattern, so you can swap them without changing the model definition.

### Regularization

Regularization is available through:

- `L1`
- `L2`

These are applied layer by layer when a layer is configured with a `regularizer`.

### Metrics

The `metrics/` package exposes:

- `Accuracy`
- `Precision`
- `Recall`
- `F1_score`

These metrics are used during training and evaluation for binary classification workflows.

### Datasets

The dataset loaders provide small synthetic classification problems:

- `Moons`
- `Circles`
- `Blobs`

They return NumPy arrays ready for training.

### Visualization

The visualization helpers include:

- Decision boundary plots
- Loss curve plots
- Network structure visualization

These tools are useful for understanding how the model behaves during training.

## Save and Load

The model container in `layers/model.py` supports serialization with pickle.

```python
model.save("moon.pkl")
restored_model = Model.load("moon.pkl")
```

This makes it easy to persist trained models and reuse them later without retraining.

## Example Workflow

A typical experiment looks like this:

1. Load a dataset from `datasets/`
2. Create a `Model`
3. Add `DenseLayer` blocks with chosen initializers and activations
4. Train with `Trainer.fit(...)`
5. Inspect metrics from `Trainer.evaluate(...)`
6. Plot the decision boundary with `visualization/decision_boundary.py`
7. Save the model with `Model.save(...)`

## Documentation Map

The project is grouped into short documentation sections:

- [docs/getting_started.md](docs/getting_started.md): setup, installation, and first training run
- [docs/layers.md](docs/layers.md): dense layers, activations, and forward/backward flow
- [docs/losses.md](docs/losses.md): loss functions and when to use them
- [docs/optimizers.md](docs/optimizers.md): optimizer behavior and training tradeoffs
- [docs/examples.md](docs/examples.md): sample models and end-to-end experiments

## Notes

- The repository currently uses `RADME.md` as the main readme file name.
- Module names follow the existing codebase casing, including `Initializer/` and `DenseLayer.py`.
- The codebase is focused on binary classification examples, but the building blocks are reusable for other small neural network experiments.
