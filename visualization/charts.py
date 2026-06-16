from __future__ import annotations

import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def stats_card(label: str, value: str, delta: str):
    return f"""
    <div class='glass-panel'>
        <div class='metric-label'>{label}</div>
        <div class='metric-value'>{value}</div>
        <div class='metric-delta'>{delta}</div>
    </div>
    """


def loss_curve(history: dict[str, list[float]]):
    epochs = np.arange(1, len(history["loss"]) + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=history["loss"], mode="lines", name="Loss", line=dict(color="#8B5CF6", width=3, shape="spline")))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(21, 21, 34, 0.82)",
        font=dict(color="#FAFAFA"),
        height=320,
        xaxis_title="Epoch",
        yaxis_title="Loss",
        xaxis=dict(gridcolor="rgba(148, 163, 184, 0.10)", zeroline=False),
        yaxis=dict(gridcolor="rgba(148, 163, 184, 0.10)", zeroline=False),
    )
    return fig


def accuracy_curve(history: dict[str, list[float]]):
    epochs = np.arange(1, len(history["accuracy"]) + 1)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=history["accuracy"],
            mode="lines",
            name="Accuracy",
            line=dict(
                color="#C4C4C4",
                width=3,
                shape="spline"
            )
        )
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(21, 21, 34, 0.82)",
        font=dict(color="#FAFAFA"),
        height=320,
        xaxis_title="Epoch",
        yaxis_title="Accuracy",

        xaxis=dict(
            gridcolor="rgba(148, 163, 184, 0.10)",
            zeroline=False
        ),

        yaxis=dict(
            range=[0, 1],
            gridcolor="rgba(18, 163, 184, 0.10)",
            zeroline=False
        )
    )

    return fig

def decision_boundary(xx, yy, zz, X, y):
    fig = go.Figure(
        data=go.Contour(
            x=xx[0],
            y=yy[:, 0],
            z=zz,
            colorscale=[
                [0.0, "#303030"],   
                [0.5, "#151522"],
                [1.0, "#4750FF"]    
            ],
            contours=dict(showlabels=False, coloring="heatmap"),
            showscale=False,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=X[:, 0],
            y=X[:, 1],
            mode="markers",
            marker=dict(
    color=y.ravel(),
    colorscale=[
                    [0.0, "#10B981"],   # High-contrast Emerald Green
                    [1.0, "#F43F5E"]
],
    size=7,
    opacity=0.9
),
            name="Samples",
        )
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        height=360,
    )
    return fig


def optimizer_comparison(results: dict[str, float]):

    names = list(results.keys())
    scores = list(results.values())

    fig = px.bar(
        x=names,
        y=scores,
        labels={
            "x": "Optimizer",
            "y": "Accuracy"
        },
        color=names,
        color_discrete_sequence=[
            "#64748B",  # SGD
            "#2563EB",  # Momentum
            "#F59E0B",  # RMSProp
            "#8B5CF6",  # Adam
        ],
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(21,21,34,0.82)",
        font=dict(color="#F8FAFC"),
        height=350,
        showlegend=False,
    )

    return fig