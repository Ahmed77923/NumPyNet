from __future__ import annotations

from pathlib import Path

import streamlit as st

from deployment.services.numpy_net_backend import (
    ACTIVATIONS,
    DATASETS,
    LOSS_FUNCTIONS,
    OPTIMIZERS,
    REGULARIZERS,
    describe_model,
    decision_boundary_predictions,
    load_model,
    save_model,
    train_model,
)
from visualization.charts import (
    stats_card,
    loss_curve,
    accuracy_curve,
    decision_boundary,
    optimizer_comparison,
)

from visualization.visualize_network import visualize_network


st.set_page_config(
    page_title="NumPyNet Studio",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    header {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        visibility: hidden;
    },

   , unsafe_allow_html=True)
    :root {
        --bg: #220707;
        --bg-2: #681717;
        --panel: rgba(150, 90, 90, 0.84);
        --panel-strong: #151522;
        --panel-border: rgba(148, 163, 184, 0.16);
        --text: #ffffff;
        --muted: #94A3B8;
        --accent: #8B5CF6;
        --accent-2: #A855F7;
        --success: #10B981;
        --warning: #F59E0B;
        --danger: #EF4444;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(139, 92, 246, 0.10), transparent 28%),
            radial-gradient(circle at right top, rgba(168, 85, 247, 0.10), transparent 26%),
            linear-gradient(180deg, #0A0A12 0%, #10101B 42%, #090910 100%);
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }

    section[data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, rgba(21, 21, 34, 0.98), rgba(10, 10, 18, 0.94));
        border-right: 1px solid rgba(148, 163, 184, 0.14);
        backdrop-filter: blur(24px);
        box-shadow: 16px 0 40px rgba(0, 0, 0, 0.28);
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    .dashboard-title {
        font-size: 2.45rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        margin-bottom: 0.25rem;
        color: var(--text);
    }

    .dashboard-subtitle {
        color: var(--muted);
        font-size: 0.95rem;
        margin-bottom: 1.2rem;
    }

    .glass-panel {
        background: var(--panel);
        border: 1px solid var(--panel-border);
        border-radius: 22px;
        padding: 1.05rem 1rem;
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.32);
        backdrop-filter: blur(22px);
    }

    .metric-label {
        color: var(--muted);
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 800;
        margin-top: 0.3rem;
        color: var(--text);
    }

    .metric-delta {
        font-size: 0.85rem;
        color: #ffffff;
    }

    .tiny-badge {
        display: inline-block;
        padding: 0.25rem 0.55rem;
        border-radius: 999px;
        border: 1px solid rgba(139, 92, 246, 0.28);
        color: #C4B5FD;
        background: rgba(139, 92, 246, 0.10);
        font-size: 0.72rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .builder-node {
        background: linear-gradient(180deg, rgba(21, 21, 34, 0.96), rgba(10, 10, 18, 0.88));
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 16px;
        padding: 0.95rem;
        text-align: center;
        box-shadow:
            inset 0 1px 0 rgba(255,255,255,0.04),
            0 10px 30px rgba(0, 0, 0, 0.18);
    }

    .builder-node strong {
        display: block;
        font-size: 0.95rem;
        margin-bottom: 0.2rem;
    }

    .builder-node span {
        color: var(--muted);
        font-size: 0.83rem;
    }

    .connector {
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(139, 92, 246, 0.95), rgba(168, 85, 247, 0.95), transparent);
        margin: 0.8rem 0;
    }

    .console-box {
        background: #0A0A12;
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 18px;
        padding: 1rem;
        font-family: 'JetBrains Mono', Consolas, monospace;
        color: #FAFAFA;
        min-height: 180px;
        white-space: pre-wrap;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
    }

    .section-heading {
        font-size: 1rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
        color: var(--text);
    }

    div[data-testid="stMetricValue"] {
        color: var(--text);
    }

    .stSelectbox label, .stSlider label, .stNumberInput label, .stRadio label {
        color: #D1D5DB !important;
        font-weight: 500;
    }

    .stButton > button {
        border-radius: 14px;
        border: 1px solid rgba(139, 92, 246, 0.22);
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.96), rgba(168, 85, 247, 0.90));
        color: #0A0A12;
        font-weight: 800;
        box-shadow: 0 16px 30px rgba(139, 92, 246, 0.14);
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        border-color: rgba(168, 85, 247, 0.35);
    }

    .stSidebar [data-testid="stRadio"] label {
        color: var(--text) !important;
    }

    .stSidebar [role="radiogroup"] {
        gap: 0.15rem;
    }

    .stSidebar [data-baseweb="radio"] > div {
        background: transparent;
    }

    .stSidebar [aria-checked="true"] + div,
    .stSidebar [aria-checked="true"] p {
        color: #C4B5FD !important;
        text-shadow: 0 0 12px rgba(139, 92, 246, 0.12);
    }

    </style>
    """,
    unsafe_allow_html=True,
)


if "training_result" not in st.session_state:
    st.session_state.training_result = None
if "model_path" not in st.session_state:
    st.session_state.model_path = str(Path.cwd() / "numpy_net_model.pkl")
if "logs" not in st.session_state:
    st.session_state.logs = []
if "summary" not in st.session_state:
    st.session_state.summary = ""


def render_sidebar():
    st.sidebar.markdown("## NumPyNet Studio")
    st.sidebar.caption("Deep learning from scratch with NumPy")
    st.sidebar.markdown("<span class='tiny-badge'>AI Lab</span>", unsafe_allow_html=True)



    st.sidebar.markdown("---")
    if st.session_state.training_result is not None:
        st.sidebar.metric("Active Model", "Trained")
        st.sidebar.metric("Dataset", st.session_state.get("dataset_name", "Moons"))
        st.sidebar.metric("Run State", "Ready")
    else:
        st.sidebar.metric("Active Model", "Untrained")
        st.sidebar.metric("Dataset", "Moons")
        st.sidebar.metric("Run State", "Idle")
    return 


def render_top_stats():
    result = st.session_state.training_result
    if result is None:
        stats = [
            ("Training Accuracy", "--", "Waiting"),
            ("Validation Accuracy", "--", "Waiting"),
            ("Current Loss", "--", "Waiting"),
            ("Epoch Progress", "0 / 0", "Waiting"),
        ]
    else:
        history = result.history
        epochs_done = len(history["loss"])
        stats = [
            ("Training Accuracy", f"{history['accuracy'][-1] * 100:.2f}%", f"{history['accuracy'][-1] * 100:.2f}%"),
            ("Validation Accuracy", f"{result.metrics['accuracy'] * 100:.2f}%", f"{result.metrics['accuracy'] * 100:.2f}%"),
            ("Current Loss", f"{history['loss'][-1]:.4f}", f"{result.metrics['loss']:.4f}"),
            ("Epoch Progress", f"{epochs_done} / {epochs_done}", "Complete"),
        ]
    cols = st.columns(4)
    for col, (label, value, delta) in zip(cols, stats):
        with col:
            st.markdown(stats_card(label, value, delta), unsafe_allow_html=True)


def render_builder_panel():
    st.markdown("<div class='section-heading'>Neural Network Builder</div>", unsafe_allow_html=True)
    cols = st.columns([1, 0.22, 1, 0.22, 1])
    with cols[0]:
        st.markdown("<div class='builder-node'><strong>Input Layer</strong><span>2 features</span></div>", unsafe_allow_html=True)
    with cols[1]:
        st.markdown("<div class='connector'></div>", unsafe_allow_html=True)
    with cols[2]:
        st.markdown("<div class='builder-node'><strong>Hidden Layers</strong><span>Dense + Activation</span></div>", unsafe_allow_html=True)
    with cols[3]:
        st.markdown("<div class='connector'></div>", unsafe_allow_html=True)
    with cols[4]:
        st.markdown("<div class='builder-node'><strong>Output Layer</strong><span>1 probability</span></div>", unsafe_allow_html=True)



def render_training_panel():
    st.markdown("<div class='section-heading'>Training Configuration</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        learning_rate = st.slider("Learning Rate", 0.0001, 0.1, 0.01, 0.0001)
        epochs = st.number_input("Epochs", min_value=10, max_value=2000, value=200, step=100)
    with c2:
        batch_size = st.selectbox("Batch Size", [8, 16, 32, 64, 128], index=2)
        loss_fn = st.selectbox("Loss Function", list(LOSS_FUNCTIONS.keys()), index=0)
    with c3:
        optimizer = st.selectbox("Optimizer", list(OPTIMIZERS.keys()), index=3)
        regularization = st.selectbox("Regularization", list(REGULARIZERS.keys()), index=0)

    regularization_strength = st.slider("Regularization Strength", 0.0001, 0.1, 0.01, 0.0001)
    hidden_layer_1 = st.slider("Hidden Layer 1 Units", 2, 64, 16)
    hidden_layer_2 = st.slider("Hidden Layer 2 Units", 2, 64, 8)
    activation = st.selectbox("Hidden Activation", ACTIVATIONS, index=0)
    output_activation = st.selectbox("Output Activation", ["sigmoid", "linear"], index=0)
    initializer = st.selectbox("Initializer", ["He", "Xavier", "Random", "Zero"], index=0)

    dataset_name = st.selectbox("Dataset", list(DATASETS.keys()), index=0)
    dataset_kwargs = {}
    if dataset_name == "Moons":
        dataset_kwargs = {"n_samples": 1000, "noise": 0.1}
    elif dataset_name == "Circles":
        dataset_kwargs = {"n_samples": 1000, "noise": 0.1}
    else:
        dataset_kwargs = {"n_samples": 1000, "centers": 2}

    train_clicked = st.button("Train Model", use_container_width=True, type="primary")
    save_clicked = st.button("Save Model", use_container_width=True)
    load_clicked = st.button("Load Model", use_container_width=True)

    return {
        "learning_rate": learning_rate,
        "epochs": int(epochs),
        "batch_size": batch_size,
        "loss_fn": loss_fn,
        "optimizer": optimizer,
        "regularization": regularization,
        "regularization_strength": regularization_strength,
        "hidden_layer_1": int(hidden_layer_1),
        "hidden_layer_2": int(hidden_layer_2),
        "activation": activation,
        "output_activation": output_activation,
        "initializer": initializer,
        "dataset_name": dataset_name,
        "dataset_kwargs": dataset_kwargs,
        "train_clicked": train_clicked,
        "save_clicked": save_clicked,
        "load_clicked": load_clicked,
    }



def render_visualization_panel():
    st.markdown("<div class='section-heading'>Visualization Area</div>", unsafe_allow_html=True)
    tab1, tab2, tab3, tab4 = st.tabs(["Decision Boundary", "Loss Curve", "Accuracy Curve",'Network Visualization']) # i will add "Optimizer Comparison" in future updates
    result = st.session_state.training_result

    with tab1:
        if result is None:
            st.info("Train a model to generate the decision boundary.")
        else:
            xx, yy, zz, X, y = decision_boundary_predictions(result.model, result.X, result.y)
            st.plotly_chart(decision_boundary(xx, yy, zz, X, y), use_container_width=True)
    with tab2:
        if result is None:
            st.info("Train a model to show the loss curve.")
        else:
            st.plotly_chart(loss_curve(result.history), use_container_width=True)
    with tab3:
        if result is None:
            st.info("Train a model to show the accuracy curve.")
        else:
            st.plotly_chart(accuracy_curve(result.history), use_container_width=True)
    with tab4:
        if result is None:
            st.info("Train a model to show the accuracy curve.")
        else:
            st.plotly_chart(visualize_network(result.model), use_container_width=True)


def render_metrics_panel():
    st.markdown("<div class='section-heading'>Metrics Section</div>", unsafe_allow_html=True)
    result = st.session_state.training_result
    cols = st.columns(4)
    if result is None:
        values = [("Accuracy", "--"), ("Precision", "--"), ("Recall", "--"), ("F1 Score", "--")]
    else:
        values = [
            ("Accuracy", f"{result.metrics['accuracy'] * 100:.2f}%"),
            ("Precision", f"{result.metrics['precision'] * 100:.2f}%"),
            ("Recall", f"{result.metrics['recall'] * 100:.2f}%"),
            ("F1 Score", f"{result.metrics['f1'] * 100:.2f}%"),
        ]
    for col, (label, value) in zip(cols, values):
        with col:
            st.metric(label, value)


def render_model_summary():
    st.markdown("<div class='section-heading'>Model Summary Panel</div>", unsafe_allow_html=True)
    result = st.session_state.training_result
    if result is None:
        empty_cols = st.columns(3)
        empty_cards = [
            ("Layers", "0", "Build and train a model"),
            ("Parameters", "0", "Weights will appear here"),
            ("Status", "Idle", "Awaiting first training run"),
        ]
        for col, (label, value, note) in zip(empty_cols, empty_cards):
            with col:
                st.markdown(
                    f"""
                    <div class='glass-panel'>
                        <div class='metric-label'>{label}</div>
                        <div class='metric-value' style='font-size:1.55rem;'>{value}</div>
                        <div class='metric-delta'>{note}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

       
        return

    summary = describe_model(result.model)
    summary_cards = st.columns(3)
    card_data = [
        ("Layers", str(summary["total_layers"]), "Trainable blocks"),
        ("Parameters", f"{summary['total_parameters']:,}", "Total weights + bias"),
        ("Output", "Binary", "Sigmoid classifier"),
    ]
    for col, (label, value, note) in zip(summary_cards, card_data):
        with col:
            st.markdown(
                f"""
                <div class='glass-panel'>
                    <div class='metric-label'>{label}</div>
                    <div class='metric-value' style='font-size:1.55rem;'>{value}</div>
                    <div class='metric-delta'>{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_console():
    st.markdown("<div class='section-heading'>Bottom Console</div>", unsafe_allow_html=True)
    result = st.session_state.training_result
    if result is None:
        logs = ["Waiting for training to start..."]
    else:
        logs = result.logs[::10]
    st.markdown(f"<div class='console-box'>{'<br>'.join(logs)}</div>", unsafe_allow_html=True)


def main():
    nav = render_sidebar()

    st.markdown("<div class='dashboard-title'>NumPyNet Studio</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='dashboard-subtitle'>A production-style deep learning dashboard for building, training, evaluating, saving, loading, and visualizing neural networks from scratch with NumPy.</div>",
        unsafe_allow_html=True,
    )

    controls = render_training_panel()
    render_top_stats()
    st.write("")

    left, right = st.columns([1.3, 1])
    with left:
        render_builder_panel()
    with right:
        # render_dataset_panel(controls["dataset_name"], controls["dataset_kwargs"])
        st.write("")
        render_metrics_panel()

    if controls["train_clicked"]:
        with st.spinner("Training real NumPyNet model..."):
            result = train_model(
                dataset_name=controls["dataset_name"],
                dataset_kwargs=controls["dataset_kwargs"],
                hidden_layers=[controls["hidden_layer_1"], controls["hidden_layer_2"]],
                initializer_name=controls["initializer"],
                activation=controls["activation"],
                output_activation=controls["output_activation"],
                loss_name=controls["loss_fn"],
                optimizer_name=controls["optimizer"],
                regularizer_name=controls["regularization"],
                regularization_strength=controls["regularization_strength"],
                epochs=controls["epochs"],
                learning_rate=controls["learning_rate"],
                log_every=max(1, controls["epochs"] // 10),
            )
            st.session_state.training_result = result
            st.session_state.logs = result.logs
            st.session_state.summary = result.summary
            st.success("Training complete.")
            st.rerun()

    result = st.session_state.training_result
    if controls["save_clicked"] and result is not None:
        save_model(result.model, st.session_state.model_path)
        st.success(f"Model saved to {st.session_state.model_path}")

    if controls["load_clicked"]:
        path = Path(st.session_state.model_path)
        if path.exists():
            loaded_model = load_model(path)
            from deployment.services.numpy_net_backend import load_dataset
            X, y = load_dataset(st.session_state.get("dataset_name", "Moons"), **controls["dataset_kwargs"])
            st.session_state.training_result = type("LoadedResult", (), {
                "model": loaded_model,
                "history": result.history if result else {"loss": [], "accuracy": [], "precision": [], "recall": [], "f1": []},
                "metrics": result.metrics if result else {"loss": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
                "summary": st.session_state.summary,
                "logs": st.session_state.logs,
                "X": X,
                "y": y,
            })()
            st.success(f"Model loaded from {path}")
        else:
            st.error(f"Model file not found: {path}")

    st.write("")
    render_visualization_panel()
    st.write("")
    render_model_summary()
    st.write("")
    render_console()

    if nav == "Settings":
        st.info("Settings can be expanded to include API keys, theme toggles, experiment logging, and export options.")


if __name__ == "__main__":
    
    main()
