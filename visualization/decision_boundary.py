import numpy as np
import matplotlib.pyplot as plt


def plot_decision_boundary(
    model,
    X,
    y,
    resolution=0.02
):

    x_min = X[:, 0].min() - 1
    x_max = X[:, 0].max() + 1

    y_min = X[:, 1].min() - 1
    y_max = X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, resolution),
        np.arange(y_min, y_max, resolution)
    )

    grid = np.c_[
        xx.ravel(),
        yy.ravel()
    ]

    # Support models that implement `predict()` or only `forward()`.
    if hasattr(model, "predict"):
        predictions = model.predict(grid)
    else:
        predictions = model.forward(grid)

    predictions = (predictions > 0.5).astype(int)

    predictions = predictions.reshape(
        xx.shape
    )

    plt.contourf(
        xx,
        yy,
        predictions,
        alpha=0.3
    )

    plt.scatter(
        X[:, 0],
        X[:, 1],
        c=y.ravel()
    )

    plt.title(
        "Decision Boundary"
    )

    plt.show()