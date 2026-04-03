# visualize.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# -----------------------------
# Utilities
# -----------------------------
def make_history_df(history):
    """
    Convert list of step dictionaries into a DataFrame.
    """
    if history is None or len(history) == 0:
        return pd.DataFrame()
    return pd.DataFrame(history)


# -----------------------------
# Training summary plots
# -----------------------------
def plot_training_curves(rewards, epsilons=None):
    """
    Plot episode rewards and epsilon decay.
    Returns matplotlib figure objects.
    """
    figs = []

    # Rewards
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(rewards, label="Episode Reward")
    ax1.set_title("Training Reward Curve")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    figs.append(fig1)

    # Epsilon
    if epsilons is not None and len(epsilons) > 0:
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(epsilons, label="Epsilon")
        ax2.set_title("Exploration Decay")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Epsilon")
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        figs.append(fig2)

    return figs


# -----------------------------
# Step-wise plots
# -----------------------------
def plot_history(history_df):
    """
    Plot irrigation actions, reward, and soil trajectory.
    """
    if history_df is None or history_df.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No history available", ha="center", va="center")
        ax.axis("off")
        return fig

    figs = []

    # Action plot
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(history_df["step"], history_df["action_mm"], label="Action (mm)")
    ax1.set_title("Irrigation Actions Over Time")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Irrigation (mm)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    figs.append(fig1)

    # Reward plot
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(history_df["step"], history_df["reward"], label="Reward")
    ax2.set_title("Reward Over Time")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Reward")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    figs.append(fig2)

    # Soil plot
    if "next_soil" in history_df.columns:
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.plot(history_df["step"], history_df["next_soil"], label="Soil Moisture")
        if "prev_soil" in history_df.columns:
            ax3.plot(history_df["step"], history_df["prev_soil"], alpha=0.5, label="Previous Soil")
        ax3.set_title("Soil Moisture Dynamics")
        ax3.set_xlabel("Step")
        ax3.set_ylabel("Soil Value")
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        figs.append(fig3)

    return figs


def plot_q_values(q_values, action_space):
    """
    Bar chart for Q-values of a single state.
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([str(a) for a in action_space], q_values)
    ax.set_title("Q-Values for Current State")
    ax.set_xlabel("Action (mm)")
    ax.set_ylabel("Q-value")
    ax.grid(True, axis="y", alpha=0.3)
    return fig


def plot_state_snapshot(state, action_mm, reward, q_values=None, feature_names=None):
    """
    Show current state as text/visual summary using matplotlib.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    if feature_names is None:
        feature_names = [
            "soil", "temp", "humidity", "wind",
            "par", "eto", "soil_norm", "moisture_gap"
        ]

    lines = []
    for i, val in enumerate(state):
        name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
        lines.append(f"{name}: {float(val):.3f}")

    lines.append(f"Selected action: {action_mm} mm")
    lines.append(f"Reward: {reward:.3f}")

    if q_values is not None:
        lines.append("Q-values: " + ", ".join([f"{v:.2f}" for v in q_values]))

    ax.text(
        0.02, 0.98,
        "\n".join(lines),
        va="top",
        ha="left",
        fontsize=11,
        family="monospace"
    )
    ax.set_title("Current Agent Snapshot")
    return fig


# -----------------------------
# Streamlit live dashboard
# -----------------------------
def render_live_dashboard(
    st,
    episode,
    step,
    state,
    action_mm,
    reward,
    total_reward,
    q_values,
    history_df=None,
    action_space=None,
):
    """
    Live update dashboard for Streamlit.
    Call this inside the training loop.
    """
    st.subheader(f"Live Agent View — Episode {episode + 1}, Step {step + 1}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Step", step + 1)
    c2.metric("Action", f"{action_mm} mm")
    c3.metric("Step Reward", f"{reward:.2f}")
    c4.metric("Episode Reward", f"{total_reward:.2f}")

    left, right = st.columns(2)

    with left:
        st.markdown("### Current State")
        feature_names = [
            "soil", "temp", "humidity", "wind",
            "par", "eto", "soil_norm", "moisture_gap"
        ]
        state_df = pd.DataFrame({
            "feature": feature_names[:len(state)],
            "value": [float(x) for x in state]
        })
        st.dataframe(state_df, use_container_width=True)

    with right:
        st.markdown("### Q-values")
        if q_values is not None and action_space is not None:
            q_df = pd.DataFrame({
                "action_mm": action_space,
                "q_value": [float(x) for x in q_values]
            })
            st.dataframe(q_df, use_container_width=True)
        else:
            st.info("Q-values not available yet.")

    if history_df is not None and not history_df.empty:
        st.markdown("### Recent Agent Behaviour")
        recent = history_df.tail(25).copy()
        st.line_chart(
            recent[["prev_soil", "next_soil"]] if "prev_soil" in recent.columns and "next_soil" in recent.columns
            else recent[["reward"]]
        )

    fig = plot_state_snapshot(state, action_mm, reward, q_values=q_values)
    st.pyplot(fig)
    plt.close(fig)