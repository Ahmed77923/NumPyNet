import plotly.graph_objects as go


def visualize_network(model):

    layers = [model.layers[0].input_size]

    for layer in model.layers:
        layers.append(layer.output_size)

    max_neurons = max(layers)

    colors = [
        "#06B6D4",  # Input
        "#8B5CF6",  # Hidden
        "#8B5CF6",
        "#EC4899"   # Output
    ]

    fig = go.Figure()

    positions = []

    x_spacing = 3

    # Neurons
    for layer_idx, neurons in enumerate(layers):

        x = layer_idx * x_spacing

        neuron_positions = []

        for neuron_idx in range(neurons):

            y = neuron_idx - (neurons - 1) / 2

            neuron_positions.append((x, y))

        positions.append(neuron_positions)

    # Connections
    for layer_idx in range(len(positions) - 1):

        current_layer = positions[layer_idx]
        next_layer = positions[layer_idx + 1]

        for x1, y1 in current_layer:
            for x2, y2 in next_layer:

                fig.add_trace(
                    go.Scatter(
                        x=[x1, x2],
                        y=[y1, y2],
                        mode="lines",
                        line=dict(
                            color="rgba(148,163,184,0.15)",
                            width=1
                        ),
                        hoverinfo="skip",
                        showlegend=False
                    )
                )

    # Nodes
    for layer_idx, neuron_positions in enumerate(positions):

        color = colors[
            min(layer_idx, len(colors) - 1)
        ]

        x_vals = [p[0] for p in neuron_positions]
        y_vals = [p[1] for p in neuron_positions]

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="markers",
                marker=dict(
                    size=24,
                    color=color,
                    line=dict(
                        color="white",
                        width=1
                    )
                ),
                hovertemplate=(
                    f"Layer {layer_idx}<br>"
                    "Neuron %{pointNumber}"
                    "<extra></extra>"
                ),
                showlegend=False
            )
        )

    # Layer labels
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

    for i, name in enumerate(layer_names):

        fig.add_annotation(
            x=i * x_spacing,
            y=max_neurons / 2 + 1.5,
            text=f"{name}<br>{layers[i]} neurons",
            showarrow=False,
            font=dict(size=13)
        )

    fig.update_layout(
        title="Neural Network Architecture",
        height=650,
        paper_bgcolor="#0A0A12",
        plot_bgcolor="#0A0A12",
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        ),
        xaxis=dict(
            visible=False
        ),
        yaxis=dict(
            visible=False,
            scaleanchor="x"
        )
    )

    return fig