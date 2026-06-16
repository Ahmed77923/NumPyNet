from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import io
import contextlib

import numpy as np

from core.DenseLayer import DenseLayer
from Initializer.He import He
from Initializer.xavier import Xavier
from Initializer.random import Random
from Initializer.zero import Zero
from layers.model import Model
from losses.mse import MSE
from losses.binary_cross_entropy import BinaryCrossEntropy
from optimizers.sgd import SGD
from optimizers.momentum import Momentum
from optimizers.rmsprop import RMSProp
from optimizers.adam import Adam
from regularization.l1 import L1
from regularization.l2 import L2
from datasets.moons import Moons
from datasets.circles import Circles
from datasets.blobs import Blobs
from core.trainer import Trainer


@dataclass
class TrainingResult:
    model: Model
    history: dict[str, list[float]]
    metrics: dict[str, float]
    summary: str
    logs: list[str]
    X: np.ndarray
    y: np.ndarray


DATASETS = {
    "Moons": Moons,
    "Circles": Circles,
    "Blobs": Blobs,
}

LOSS_FUNCTIONS = {
    "MSE": MSE,
    "Binary Cross Entropy": BinaryCrossEntropy,
}

OPTIMIZERS = {
    "SGD": SGD,
    "Momentum": Momentum,
    "RMSProp": RMSProp,
    "Adam": Adam,
}

REGULARIZERS = {
    "None": None,
    "L1": L1,
    "L2": L2,
}

INITIALIZERS = {
    "He": He,
    "Xavier": Xavier,
    "Random": Random,
    "Zero": Zero,
}

ACTIVATIONS = ["relu", "sigmoid", "tanh", "linear"]


class TrainingLogger:
    def __init__(self):
        self.lines: list[str] = []

    def append(self, epoch: int, loss: float, accuracy: float) -> None:
        self.lines.append(
            f"Epoch {epoch:04d} | Loss {loss:.6f} | Accuracy {accuracy:.2%}"
        )


def load_dataset(name: str, **kwargs):
    loader = DATASETS[name](**kwargs)
    return loader.load()


def build_regularizer(name: str, regularization_strength: float):
    regularizer_cls = REGULARIZERS[name]
    if regularizer_cls is None:
        return None
    return regularizer_cls(regularization_strength)


def build_loss(name: str):
    return LOSS_FUNCTIONS[name]()


def build_optimizer(name: str):
    return OPTIMIZERS[name]()


def build_model(
    input_size: int,
    hidden_layers: list[int],
    output_size: int,
    initializer_name: str,
    activation: str,
    output_activation: str,
    regularizer,
):
    model = Model()
    layer_sizes = [input_size] + hidden_layers + [output_size]
    activations = [activation] * len(hidden_layers) + [output_activation]

    for idx in range(len(layer_sizes) - 1):
        initializer = INITIALIZERS[initializer_name]()
        model.add(
            DenseLayer(
                layer_sizes[idx],
                layer_sizes[idx + 1],
                initializer,
                activations[idx],
                regularizer=regularizer,
            )
        )

    return model


def summarize_model(model: Model) -> str:
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        model.summary()
    return buffer.getvalue()


def describe_model(model: Model):
    rows = []
    total_params = 0

    for index, layer in enumerate(getattr(model, "layers", []), start=1):
        params = int(layer.weights.size + layer.bias.size)
        total_params += params

        activation_name = getattr(getattr(layer, "activation", None), "__class__", type("Unknown", (), {})).__name__
        regularizer = getattr(layer, "regularizer", None)
        regularizer_name = regularizer.__class__.__name__ if regularizer is not None else "None"

        rows.append(
            {
                "index": index,
                "layer": layer.__class__.__name__,
                "input_shape": f"(None, {layer.input_size})",
                "output_shape": f"(None, {layer.output_size})",
                "activation": activation_name,
                "regularizer": regularizer_name,
                "parameters": params,
            }
        )

    return {
        "rows": rows,
        "total_layers": len(rows),
        "total_parameters": total_params,
    }


def train_model(
    dataset_name: str,
    dataset_kwargs: dict[str, Any],
    hidden_layers: list[int],
    initializer_name: str,
    activation: str,
    output_activation: str,
    loss_name: str,
    optimizer_name: str,
    regularizer_name: str,
    regularization_strength: float,
    epochs: int,
    learning_rate: float,
    log_every: int,
):
    X, y = load_dataset(dataset_name, **dataset_kwargs)
    regularizer = build_regularizer(regularizer_name, regularization_strength)
    model = build_model(
        input_size=X.shape[1],
        hidden_layers=hidden_layers,
        output_size=1,
        initializer_name=initializer_name,
        activation=activation,
        output_activation=output_activation,
        regularizer=regularizer,
    )

    loss_fn = build_loss(loss_name)
    optimizer = build_optimizer(optimizer_name)
    trainer = Trainer(model=model, loss_fn=loss_fn, optimizer=optimizer, learning_rate=learning_rate)

    logger = TrainingLogger()
    history = {"loss": [], "accuracy": [], "precision": [], "recall": [], "f1": []}

    for epoch in range(1, epochs + 1):
        pred = model.forward(X)
        loss = loss_fn.forward(y, pred)
        grad = loss_fn.backward(y, pred)
        model.backward(grad, learning_rate, optimizer)

        regularization_loss = 0.0
        for layer in getattr(model, "layers", []):
            regularizer_obj = getattr(layer, "regularizer", None)
            if regularizer_obj is not None:
                regularization_loss += regularizer_obj.penalty(layer.weights)

        total_loss = float(loss + regularization_loss)
        metrics = trainer.evaluate(X, y)

        history["loss"].append(total_loss)
        history["accuracy"].append(float(metrics["accuracy"]))
        history["precision"].append(float(metrics["precision"]))
        history["recall"].append(float(metrics["recall"]))
        history["f1"].append(float(metrics["f1"]))

        logger.append(epoch, total_loss, float(metrics["accuracy"]))

        if log_every and epoch % log_every == 0:
            pass

    final_metrics = trainer.evaluate(X, y)
    summary = summarize_model(model)

    return TrainingResult(
        model=model,
        history=history,
        metrics={k: float(v) for k, v in final_metrics.items()},
        summary=summary,
        logs=logger.lines,
        X=X,
        y=y,
    )


def decision_boundary_predictions(model: Model, X: np.ndarray, y: np.ndarray, resolution: float = 0.02):
    x_min = X[:, 0].min() - 1
    x_max = X[:, 0].max() + 1
    y_min = X[:, 1].min() - 1
    y_max = X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, resolution),
        np.arange(y_min, y_max, resolution),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    predictions = model.forward(grid)
    if predictions.ndim > 1 and predictions.shape[1] > 1:
        predictions = predictions[:, :1]
    return xx, yy, predictions.reshape(xx.shape), X, y


def save_model(model: Model, path: str | Path):
    model.save(str(path))


def load_model(path: str | Path):
    return Model.load(str(path))
