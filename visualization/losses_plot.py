import matplotlib.pyplot as plt


def plot_loss_curve(
    losses,
    title="Training Loss"
):

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        losses,
        linewidth=2,
        label="Loss"
    )

    plt.title(
        title,
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel(
        "Epoch",
        fontsize=12
    )

    plt.ylabel(
        "Loss",
        fontsize=12
    )

    plt.grid(
        True,
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    plt.show()