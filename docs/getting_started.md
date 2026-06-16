# Getting Started

NumPyNet is a small neural-network library built on NumPy. It is designed for simple feed-forward experiments, binary classification, and quick iteration on custom optimizers, initializers, and visualizations.

## Installation

Install the runtime dependencies first:

```bash
pip install numpy scikit-learn matplotlib
```

## First Training Run

A typical workflow looks like this:

1. Load a dataset from `datasets/`
2. Build a `Model`
3. Add `DenseLayer` blocks
4. Train with `Trainer.fit(...)`
5. Evaluate with `Trainer.evaluate(...)`
6. Plot the decision boundary or loss curve
7. Save the model if needed

### Example

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

trainer = Trainer(model, MSE(), Adam(), learning_rate=0.01)
history = trainer.fit(X, y, epochs=5000)
metrics = trainer.evaluate(X, y)

print(history)
print(metrics)
```

## Saving Models

The model container in `layers/model.py` supports serialization with pickle:

```python
model.save("moon.pkl")
restored_model = Model.load("moon.pkl")
```

## Suggested Next Steps

- Read [layers.md](layers.md) for layer and model details
- Read [losses.md](losses.md) for loss functions
- Read [optimizers.md](optimizers.md) for training updates
- Read [examples.md](examples.md) for end-to-end experiments
