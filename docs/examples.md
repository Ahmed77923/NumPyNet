# Examples

This project is centered around small, readable experiments. The current codebase includes a training demo in `core/trainer.py` that trains a dense network on the moons dataset, evaluates several metrics, and plots the decision boundary.

## Moons Classification Example

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

print(history)
print(metrics)
```

## Save and Reload

```python
model.save("moon.pkl")
restored_model = Model.load("moon.pkl")
```

## Visualize the Boundary

```python
from visualization.decision_boundary import plot_decision_boundary

plot_decision_boundary(model, X, y, resolution=0.02)
```

## Other Datasets

You can swap the dataset loader without changing the rest of the pipeline:

- `Circles(n_samples=1000, noise=0.1).load()`
- `Blobs(n_samples=1000, centers=2).load()`

## Experiment Ideas

- Compare `Adam` against `SGD`, `Momentum`, and `RMSProp`
- Swap `He` and `Xavier` initializers
- Add `L1` or `L2` regularization to individual layers
- Replace `MSE` with binary cross entropy for classification
