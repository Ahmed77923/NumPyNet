import matplotlib.pyplot as plt


def visualize_network(model):

    # Build network shape
    layers = [model.layers[0].input_size]

    for layer in model.layers:
        layers.append(layer.output_size)

    max_neurons = max(layers)

    fig, ax = plt.subplots(figsize=(14, 8))

    x_spacing = 4

    colors = [
        "#3B82F6",  # Input
        "#10B981",  # Hidden
        "#10B981",
        "#EF4444"   # Output
    ]

    layer_names = []

    for i in range(len(layers)):

        if i == 0:
            layer_names.append("Input")

        elif i == len(layers) - 1:
            layer_names.append("Output")

        else:
            layer_names.append(
                f"Hidden {i}"
            )

    positions = []

    # Draw neurons
    for layer_idx, neurons in enumerate(layers):

        x = layer_idx * x_spacing

        color = colors[
            min(layer_idx, len(colors) - 1)
        ]

        neuron_positions = []

        for neuron_idx in range(neurons):

            y = (
                neuron_idx
                - (neurons - 1) / 2
            )

            neuron_positions.append(
                (x, y)
            )

            circle = plt.Circle(
                (x, y),
                0.25,
                color=color,
                ec="black",
                lw=1.5
            )

            ax.add_patch(circle)

        positions.append(
            neuron_positions
        )

        # Layer title
        ax.text(
            x,
            max_neurons / 2 + 1,
            layer_names[layer_idx],
            ha="center",
            fontsize=13,
            fontweight="bold"
        )

        # Neuron count
        ax.text(
            x,
            -(max_neurons / 2) - 1,
            f"{neurons} neurons",
            ha="center",
            fontsize=10
        )

    # Draw connections
    for layer_idx in range(
        len(positions) - 1
    ):

        current_layer = positions[
            layer_idx
        ]

        next_layer = positions[
            layer_idx + 1
        ]

        for x1, y1 in current_layer:

            for x2, y2 in next_layer:

                ax.plot(
                    [x1, x2],
                    [y1, y2],
                    color="gray",
                    alpha=0.08,
                    linewidth=0.8
                )

    plt.title(
        "Neural Network Architecture",
        fontsize=18,
        fontweight="bold",
        pad=20
    )

    ax.set_aspect("equal")
    ax.axis("off")

    plt.tight_layout()

    plt.show()