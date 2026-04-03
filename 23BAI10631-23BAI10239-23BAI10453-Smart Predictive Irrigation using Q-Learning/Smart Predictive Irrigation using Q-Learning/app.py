# app.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from preprocess import load_data
from environment import IrrigationEnv
from rl_agent import QLearningAgent


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Smart Irrigation RL",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌱 Smart Irrigation using Q-Learning")
st.caption("Fast training dashboard + full episode-step logs + saved-model predictive planner")


# -----------------------------
# Paths
# -----------------------------
ARTIFACT_DIR = "artifacts"
SAVE_PATH = os.path.join(ARTIFACT_DIR, "q_table.pkl")


# -----------------------------
# Helpers
# -----------------------------
def make_history_df(history):
    if history is None or len(history) == 0:
        return pd.DataFrame()
    return pd.DataFrame(history)


def clamp(x, low, high):
    return max(low, min(high, x))


@st.cache_data(show_spinner=False)
def cached_load_data(sensor_path, daily_path, use_daily_average):
    return load_data(
        sensor_path=sensor_path,
        daily_path=daily_path,
        use_daily_average=use_daily_average,
        add_rl_bins=False,
    )


def plot_training_curves(rewards, epsilons=None):
    figs = []

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(rewards)
    ax1.set_title("Episode Reward Curve")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.grid(True, alpha=0.3)
    figs.append(fig1)

    if epsilons is not None and len(epsilons) > 0:
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        ax2.plot(epsilons)
        ax2.set_title("Epsilon Decay")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Epsilon")
        ax2.grid(True, alpha=0.3)
        figs.append(fig2)

    return figs


def plot_history(history_df):
    figs = []
    if history_df is None or history_df.empty:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.text(0.5, 0.5, "No history yet", ha="center", va="center")
        ax.axis("off")
        return [fig]

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(history_df["step"], history_df["action_mm"])
    ax1.set_title("Irrigation Actions Over Time")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Action (mm)")
    ax1.grid(True, alpha=0.3)
    figs.append(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(history_df["step"], history_df["reward"])
    ax2.set_title("Reward Over Time")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Reward")
    ax2.grid(True, alpha=0.3)
    figs.append(fig2)

    if "prev_soil" in history_df.columns and "next_soil" in history_df.columns:
        fig3, ax3 = plt.subplots(figsize=(10, 4))
        ax3.plot(history_df["step"], history_df["prev_soil"], label="Previous Soil")
        ax3.plot(history_df["step"], history_df["next_soil"], label="Next Soil")
        ax3.set_title("Soil Moisture Dynamics")
        ax3.set_xlabel("Step")
        ax3.set_ylabel("Soil")
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        figs.append(fig3)

    return figs


def plot_q_values(q_values, action_space):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([str(a) for a in action_space], q_values)
    ax.set_title("Q-values for Current State")
    ax.set_xlabel("Action (mm)")
    ax.set_ylabel("Q-value")
    ax.grid(True, axis="y", alpha=0.3)
    return fig


def plot_state_snapshot(state, action_mm, reward, q_values=None):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axis("off")

    feature_names = [
        "soil", "temp", "humidity", "wind", "par",
        "eto", "rain", "soil_norm", "moisture_gap",
        "demand_pressure", "climate_stress"
    ]

    lines = []
    if state is not None and len(state) > 0:
        for i, val in enumerate(state):
            name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
            try:
                lines.append(f"{name}: {float(val):.3f}")
            except Exception:
                lines.append(f"{name}: {val}")

    lines.append(f"selected action: {action_mm} mm")
    lines.append(f"step reward: {reward:.3f}")

    if q_values is not None:
        try:
            lines.append("q-values: " + ", ".join([f"{float(v):.2f}" for v in q_values]))
        except Exception:
            pass

    ax.text(
        0.02, 0.98, "\n".join(lines),
        ha="left", va="top",
        fontsize=11, family="monospace"
    )
    ax.set_title("Current Agent Snapshot")
    return fig


def render_live_dashboard(
    episode,
    step,
    state,
    action_mm,
    reward,
    total_reward,
    q_values,
    history_df,
    action_space,
):
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
            "soil", "temp", "humidity", "wind", "par",
            "eto", "rain", "soil_norm", "moisture_gap",
            "demand_pressure", "climate_stress"
        ]
        if state is not None and len(state) > 0:
            state_df = pd.DataFrame({
                "feature": feature_names[:len(state)],
                "value": [float(x) for x in state]
            })
            st.dataframe(state_df, use_container_width=True, hide_index=True)
        else:
            st.info("State not available.")

    with right:
        st.markdown("### Q-values")
        if q_values is not None and action_space is not None:
            q_df = pd.DataFrame({
                "action_mm": action_space,
                "q_value": [float(x) for x in q_values]
            })
            st.dataframe(q_df, use_container_width=True, hide_index=True)
        else:
            st.info("Q-values not available yet.")

    if history_df is not None and not history_df.empty:
        st.markdown("### Recent Trend")
        recent = history_df.tail(30).copy()
        cols = [c for c in ["prev_soil", "next_soil", "reward"] if c in recent.columns]
        if len(cols) > 0:
            st.line_chart(recent[cols], use_container_width=True)

    fig = plot_state_snapshot(state, action_mm, reward, q_values=q_values)
    st.pyplot(fig)
    plt.close(fig)


def simple_baseline_evaluation(
    env_class,
    data,
    action_space,
    target_low,
    target_high,
    eto_noise_std,
    rain_noise_std,
    reward_scale,
    baseline_action=0,
    max_steps=None,
):
    env = env_class(
        data,
        action_space=action_space,
        target_soil_range=(target_low, target_high),
        eto_noise_std=eto_noise_std,
        rain_noise_std=rain_noise_std,
        reward_scale=reward_scale,
        seed=7,
    )
    _ = env.reset()
    total_reward = 0.0
    total_water = 0.0
    steps = 0

    while True:
        _, reward, done, info = env.step(baseline_action)
        total_reward += reward
        total_water += info["action_mm"]
        steps += 1
        if done or (max_steps is not None and steps >= max_steps):
            break

    return {
        "total_reward": float(total_reward),
        "total_water": float(total_water),
        "steps": steps,
    }


def load_saved_agent():
    if not os.path.exists(SAVE_PATH):
        raise FileNotFoundError(f"Saved agent not found at: {SAVE_PATH}")
    return QLearningAgent.load(SAVE_PATH)


def estimate_starting_soil(current_soil, last_irrigation_mm, days_since_last, eto):
    wetting_effect = 0.20 * float(last_irrigation_mm)
    drying_effect = float(days_since_last) * max(0.5, 0.08 * float(eto))
    return current_soil + wetting_effect - drying_effect


def build_10_day_forecast_from_user(
    starting_soil,
    temp,
    humidity,
    wind,
    par,
    eto,
    rain_prob,
    days=10,
    temp_trend=0.25,
):
    rows = []
    rng = np.random.default_rng(42)

    for d in range(days):
        day_temp = temp + d * temp_trend + rng.normal(0, 0.7)
        day_humidity = clamp(humidity - d * 0.4 + rng.normal(0, 2.0), 0, 100)
        day_wind = max(0.0, wind + rng.normal(0, 0.4))
        day_par = max(0.0, par + rng.normal(0, max(0.05 * max(par, 1e-6), 0.02)))
        day_eto = max(0.0, eto + d * 0.12 + rng.normal(0, 0.35))
        day_rain_prob = clamp(rain_prob + rng.normal(0, 0.08), 0, 1)

        rows.append({
            "soil": float(starting_soil),
            "temp": float(day_temp),
            "humidity": float(day_humidity),
            "wind": float(day_wind),
            "par": float(day_par),
            "eto": float(day_eto),
            "rain_prob": float(day_rain_prob),
        })

    return pd.DataFrame(rows)


def assign_time_slot(action_mm, day_temp, weather_type):
    if action_mm <= 0:
        return "No irrigation"

    if weather_type in ["Rainy", "Cloudy", "Humid"]:
        if action_mm <= 5:
            return "Morning"
        elif action_mm <= 10:
            return "Evening"
        else:
            return "Night"

    if day_temp >= 35:
        if action_mm <= 5:
            return "Evening"
        elif action_mm <= 10:
            return "Night"
        else:
            return "Midnight"

    if day_temp >= 28:
        if action_mm <= 5:
            return "Morning"
        elif action_mm <= 10:
            return "Evening"
        else:
            return "Night"

    if action_mm <= 5:
        return "Morning"
    elif action_mm <= 10:
        return "Afternoon"
    else:
        return "Evening"


def run_10_day_schedule(
    agent,
    forecast_df,
    action_space,
    target_low,
    target_high,
    starting_soil,
    field_area_ha,
    weather_type,
    eto_noise_std=0.8,
    rain_noise_std=0.5,
    reward_scale=1.0,
):
    planner_env = IrrigationEnv(
        data=forecast_df,
        action_space=action_space,
        target_soil_range=(target_low, target_high),
        eto_noise_std=eto_noise_std,
        rain_noise_std=rain_noise_std,
        reward_scale=reward_scale,
        seed=42,
    )

    _ = planner_env.reset()
    planner_env.current_soil = float(starting_soil)

    first_state = planner_env._get_state()
    day1_policy = agent.policy_summary(first_state)

    schedule = []
    total_mm = 0.0
    total_liters = 0.0

    for day in range(len(forecast_df)):
        state = planner_env._get_state()
        action_idx = agent.predict_action(state)
        _, reward, done, info = planner_env.step(action_idx)

        action_mm = float(info["action_mm"])
        liters = action_mm * float(field_area_ha) * 10000.0

        total_mm += action_mm
        total_liters += liters

        day_temp = float(forecast_df.iloc[day]["temp"])
        time_slot = assign_time_slot(action_mm, day_temp, weather_type)

        schedule.append({
            "day": day + 1,
            "time": time_slot,
            "soil_before": round(float(info["prev_soil"]), 2),
            "eto": round(float(info["eto"]), 2),
            "rain": round(float(info["rain"]), 2),
            "action_mm": round(action_mm, 2),
            "water_liters": round(liters, 2),
            "soil_after": round(float(info["next_soil"]), 2),
            "reward": round(float(reward), 2),
        })

        if done:
            break

    return pd.DataFrame(schedule), planner_env.get_history_df(), total_mm, total_liters, day1_policy


def train_agent_fast(
    env,
    agent,
    episodes,
    log_all_steps=True,
    live_enabled=False,
    live_every_episode=10,
    live_every_step=20,
):
    training_rewards = []
    training_eps = []
    episode_step_rows = []
    live_payload = None

    for ep in range(episodes):
        state = env.reset()
        total_reward = 0.0
        step = 0
        done = False

        agent.start_episode(ep)

        while not done:
            action_idx, choose_info = agent.choose_action(state, training=True, return_info=True)
            next_state, reward, done, env_info = env.step(action_idx)
            new_q = agent.update(state, action_idx, reward, next_state, done=done)

            agent.track_episode_step(reward)
            total_reward += reward

            row = {
                "episode": ep + 1,
                "step": step + 1,
                "action_idx": int(action_idx),
                "action_mm": float(env_info["action_mm"]),
                "reward": float(reward),
                "total_reward_so_far": float(total_reward),
                "epsilon": float(agent.epsilon),
                "state_seen": bool(choose_info.get("state_seen", False)),
                "explored": bool(choose_info.get("explored", False)),
                "q_updated_value": float(new_q),
                "prev_soil": float(env_info.get("prev_soil", np.nan)),
                "next_soil": float(env_info.get("next_soil", np.nan)),
                "eto": float(env_info.get("eto", np.nan)),
                "eto_noisy": float(env_info.get("eto_noisy", np.nan)),
                "rain": float(env_info.get("rain", np.nan)),
                "rain_noisy": float(env_info.get("rain_noisy", np.nan)),
                "natural_loss": float(env_info.get("natural_loss", np.nan)),
                "effective_irrigation": float(env_info.get("effective_irrigation", np.nan)),
                "drainage": float(env_info.get("drainage", np.nan)),
                "soil_norm": float(env_info.get("soil_norm", np.nan)),
                "moisture_gap": float(env_info.get("moisture_gap", np.nan)),
                "done": bool(done),
            }

            if log_all_steps:
                episode_step_rows.append(row)

            if live_enabled and (ep % live_every_episode == 0) and ((step % live_every_step == 0) or done):
                agent.record_step(
                    episode=ep,
                    step=step,
                    state=state,
                    action_idx=action_idx,
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    info=env_info,
                )
                live_payload = {
                    "episode": ep,
                    "step": step,
                    "state": next_state if next_state is not None else state,
                    "action_mm": env_info["action_mm"],
                    "reward": reward,
                    "total_reward": total_reward,
                    "q_values": agent.get_q_values(next_state if next_state is not None else state),
                    "history_df": make_history_df(agent.training_log),
                }

            state = next_state
            step += 1

        summary = agent.end_episode()
        agent.decay_epsilon()

        training_rewards.append(total_reward)
        training_eps.append(agent.epsilon)

    episode_steps_df = pd.DataFrame(episode_step_rows)
    return training_rewards, training_eps, live_payload, episode_steps_df


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Settings")

sensor_path = st.sidebar.text_input(
    "All-Data-SensorParser.xlsx path",
    value=r"C:\Users\LENOVO\Desktop\All-Data-SensorParser.xlsx"
)

daily_path = st.sidebar.text_input(
    "DailyAverageSensedData1.xlsx path",
    value=r"C:\Users\LENOVO\Desktop\DailyAverageSensedData1.xlsx"
)

use_daily_average = st.sidebar.checkbox("Use daily average file first", value=True)
train_episodes = st.sidebar.slider("Training episodes", 10, 500, 200)
action_max = st.sidebar.select_slider("Max irrigation action (mm)", options=[10, 15, 20, 25, 30, 40], value=20)
action_step = st.sidebar.select_slider("Action step (mm)", options=[1, 2, 5], value=5)

lr = st.sidebar.slider("Learning rate", 0.01, 0.5, 0.1, 0.01)
gamma = st.sidebar.slider("Discount factor", 0.5, 0.99, 0.95, 0.01)
epsilon = st.sidebar.slider("Initial epsilon", 0.01, 1.0, 1.0, 0.01)
epsilon_min = st.sidebar.slider("Minimum epsilon", 0.01, 0.5, 0.10, 0.01)
epsilon_decay = st.sidebar.slider("Epsilon decay", 0.90, 0.9999, 0.999, 0.0001)

use_discretization = st.sidebar.checkbox("Use discretization", value=True)
n_bins = st.sidebar.slider("Bins per feature", 4, 20, 8)

target_low = st.sidebar.slider("Target soil lower bound", 0.0, 100.0, 35.0, 1.0)
target_high = st.sidebar.slider("Target soil upper bound", 0.0, 100.0, 70.0, 1.0)
eto_noise_std = st.sidebar.slider("ETo noise std", 0.0, 5.0, 0.8, 0.1)
rain_noise_std = st.sidebar.slider("Rain noise std", 0.0, 5.0, 0.5, 0.1)

reward_scale = st.sidebar.slider("Reward scale", 0.1, 5.0, 1.0, 0.1)

fast_mode = st.sidebar.checkbox("Fast mode training", value=True)
show_live_training = st.sidebar.checkbox("Show live dashboard during training", value=False)
log_all_episode_steps = st.sidebar.checkbox("Log every step of every episode", value=True)
live_every_episode = st.sidebar.slider("Live update every N episodes", 1, 50, 10)
live_every_step = st.sidebar.slider("Live update every N steps", 1, 50, 20)

train_button = st.sidebar.button("Load Data + Train")
save_button = st.sidebar.button("Save Agent")
reset_button = st.sidebar.button("Reset Session")


# -----------------------------
# Session state
# -----------------------------
defaults = {
    "data": None,
    "env": None,
    "agent": None,
    "training_rewards": [],
    "training_eps": [],
    "trained": False,
    "last_history_df": pd.DataFrame(),
    "baseline_results": None,
    "episode_steps_df": pd.DataFrame(),
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if reset_button:
    for k in list(defaults.keys()):
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()


# -----------------------------
# Load data
# -----------------------------
if train_button or st.session_state.data is None:
    try:
        data = cached_load_data(sensor_path, daily_path, use_daily_average)
        st.session_state.data = data
        st.success(f"Data loaded successfully. Rows: {len(data)}, Columns: {len(data.columns)}")
        st.dataframe(data.head(10), use_container_width=True, hide_index=True)
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

data = st.session_state.data

if data is None or len(data) == 0:
    st.warning("No data available. Please load the Excel files.")
    st.stop()


# -----------------------------
# Build action space
# -----------------------------
actions = list(range(0, action_max + 1, action_step))
if actions[0] != 0:
    actions = [0] + actions
actions = sorted(list(dict.fromkeys(actions)))


# -----------------------------
# Initialize environment / agent
# -----------------------------
if st.session_state.env is None or train_button:
    st.session_state.env = IrrigationEnv(
        data=data,
        action_space=actions,
        target_soil_range=(target_low, target_high),
        eto_noise_std=eto_noise_std,
        rain_noise_std=rain_noise_std,
        reward_scale=reward_scale,
        seed=42,
    )

if st.session_state.agent is None or train_button:
    agent = QLearningAgent(
        action_space=actions,
        lr=lr,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        use_discretization=use_discretization,
        n_bins=n_bins,
        seed=42,
    )

    if use_discretization:
        feature_bounds = [
            (float(data["soil"].min()), float(data["soil"].max())),
            (float(data["temp"].min()), float(data["temp"].max())) if "temp" in data.columns else (0, 50),
            (float(data["humidity"].min()), float(data["humidity"].max())) if "humidity" in data.columns else (0, 100),
            (float(data["wind"].min()), float(data["wind"].max())) if "wind" in data.columns else (0, 20),
            (float(data["par"].min()), float(data["par"].max())) if "par" in data.columns else (0, 1000),
            (float(data["eto"].min()), float(data["eto"].max())) if "eto" in data.columns else (0, 20),
            (0.0, 50.0),
            (0.0, 1.0),
            (-1000.0, 1000.0),
            (0.0, 100.0),
            (0.0, 100.0),
        ]
        agent.set_feature_bounds(feature_bounds)

    st.session_state.agent = agent

env = st.session_state.env
agent = st.session_state.agent

st.sidebar.markdown("---")
st.sidebar.write(f"Actions: {actions}")
st.sidebar.write(f"Q-table size: {agent.q_table_size() if agent is not None else 0}")


# -----------------------------
# Train
# -----------------------------
if train_button:
    agent.reset_logs()
    st.session_state.training_rewards = []
    st.session_state.training_eps = []
    st.session_state.episode_steps_df = pd.DataFrame()

    progress = st.progress(0.0)
    status = st.empty()
    live_box = st.container()

    if fast_mode:
        rewards, eps, live_payload, episode_steps_df = train_agent_fast(
            env=env,
            agent=agent,
            episodes=train_episodes,
            log_all_steps=log_all_episode_steps,
            live_enabled=show_live_training,
            live_every_episode=max(1, live_every_episode),
            live_every_step=max(1, live_every_step),
        )

        st.session_state.training_rewards = rewards
        st.session_state.training_eps = eps
        st.session_state.episode_steps_df = episode_steps_df

        progress.progress(1.0)
        status.write(
            f"Fast training completed | Episodes: {train_episodes} | "
            f"Last reward: {rewards[-1]:.2f} | "
            f"Epsilon: {eps[-1]:.3f}"
        )

        if show_live_training and live_payload is not None:
            with live_box:
                render_live_dashboard(
                    episode=live_payload["episode"],
                    step=live_payload["step"],
                    state=live_payload["state"],
                    action_mm=live_payload["action_mm"],
                    reward=live_payload["reward"],
                    total_reward=live_payload["total_reward"],
                    q_values=live_payload["q_values"],
                    history_df=live_payload["history_df"],
                    action_space=actions,
                )

    else:
        episode_rows = []

        for ep in range(train_episodes):
            state = env.reset()
            total_reward = 0.0
            step = 0
            done = False

            agent.start_episode(ep)

            while not done:
                action_idx, choose_info = agent.choose_action(state, training=True, return_info=True)
                next_state, reward, done, env_info = env.step(action_idx)
                new_q = agent.update(state, action_idx, reward, next_state, done=done)
                agent.track_episode_step(reward)

                total_reward += reward

                row = {
                    "episode": ep + 1,
                    "step": step + 1,
                    "action_idx": int(action_idx),
                    "action_mm": float(env_info["action_mm"]),
                    "reward": float(reward),
                    "total_reward_so_far": float(total_reward),
                    "epsilon": float(agent.epsilon),
                    "state_seen": bool(choose_info.get("state_seen", False)),
                    "explored": bool(choose_info.get("explored", False)),
                    "q_updated_value": float(new_q),
                    "prev_soil": float(env_info.get("prev_soil", np.nan)),
                    "next_soil": float(env_info.get("next_soil", np.nan)),
                    "eto": float(env_info.get("eto", np.nan)),
                    "eto_noisy": float(env_info.get("eto_noisy", np.nan)),
                    "rain": float(env_info.get("rain", np.nan)),
                    "rain_noisy": float(env_info.get("rain_noisy", np.nan)),
                    "natural_loss": float(env_info.get("natural_loss", np.nan)),
                    "effective_irrigation": float(env_info.get("effective_irrigation", np.nan)),
                    "drainage": float(env_info.get("drainage", np.nan)),
                    "soil_norm": float(env_info.get("soil_norm", np.nan)),
                    "moisture_gap": float(env_info.get("moisture_gap", np.nan)),
                    "done": bool(done),
                }
                if log_all_episode_steps:
                    episode_rows.append(row)

                if show_live_training and ((step % live_every_step == 0) or done):
                    agent.record_step(
                        episode=ep,
                        step=step,
                        state=state,
                        action_idx=action_idx,
                        reward=reward,
                        next_state=next_state,
                        done=done,
                        info=env_info,
                    )
                    history_df = make_history_df(agent.training_log)
                    q_values = agent.get_q_values(next_state if next_state is not None else state)
                    with live_box:
                        render_live_dashboard(
                            episode=ep,
                            step=step,
                            state=next_state if next_state is not None else state,
                            action_mm=env_info["action_mm"],
                            reward=reward,
                            total_reward=total_reward,
                            q_values=q_values,
                            history_df=history_df,
                            action_space=actions,
                        )

                state = next_state
                step += 1

            summary = agent.end_episode()
            agent.decay_epsilon()

            st.session_state.training_rewards.append(total_reward)
            st.session_state.training_eps.append(agent.epsilon)

            progress.progress((ep + 1) / train_episodes)
            status.write(
                f"Episode {ep + 1}/{train_episodes} | "
                f"Reward: {total_reward:.2f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Steps: {summary['steps']}"
            )

        st.session_state.episode_steps_df = pd.DataFrame(episode_rows)

    st.session_state.trained = True
    st.session_state.last_history_df = make_history_df(agent.training_log)

    try:
        st.session_state.baseline_results = simple_baseline_evaluation(
            IrrigationEnv,
            data=data,
            action_space=actions,
            target_low=target_low,
            target_high=target_high,
            eto_noise_std=eto_noise_std,
            rain_noise_std=rain_noise_std,
            reward_scale=reward_scale,
            baseline_action=0,
        )
    except Exception as e:
        st.session_state.baseline_results = {"error": str(e)}

    st.success("Training completed successfully!")


# -----------------------------
# Summary Metrics
# -----------------------------
st.subheader("Project Summary")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", len(data))
c2.metric("Columns", len(data.columns))
c3.metric("Actions", len(actions))
c4.metric("Q-table states", agent.q_table_size())

if st.session_state.baseline_results is not None and "error" not in st.session_state.baseline_results:
    b = st.session_state.baseline_results
    st.info(
        f"Baseline (always 0 mm) — total reward: {b['total_reward']:.2f}, "
        f"water used: {b['total_water']:.2f}, steps: {b['steps']}"
    )


# -----------------------------
# Visualizations
# -----------------------------
if st.session_state.trained and len(st.session_state.training_rewards) > 0:
    st.markdown("## Training Results")

    left, right = st.columns(2)
    with left:
        for fig in plot_training_curves(st.session_state.training_rewards, st.session_state.training_eps):
            st.pyplot(fig)
            plt.close(fig)

    with right:
        if st.session_state.last_history_df is not None and not st.session_state.last_history_df.empty:
            for fig in plot_history(st.session_state.last_history_df):
                st.pyplot(fig)
                plt.close(fig)

    st.markdown("## Agent Behaviour Snapshot")

    if env.index < env.max_steps:
        try:
            state_for_q = env._get_state()
            qvals = agent.get_q_values(state_for_q)
            fig = plot_q_values(qvals, actions)
            st.pyplot(fig)
            plt.close(fig)
        except Exception:
            pass


# -----------------------------
# Full Episode-Step Table
# -----------------------------
st.markdown("---")
st.header("📋 Episode-wise Detailed Action Log")

episode_steps_df = st.session_state.episode_steps_df

if episode_steps_df is not None and not episode_steps_df.empty:
    col1, col2, col3 = st.columns(3)

    with col1:
        available_episodes = sorted(episode_steps_df["episode"].unique().tolist())
        selected_episode = st.selectbox("Select Episode", available_episodes, index=len(available_episodes) - 1)

    with col2:
        show_only_explored = st.checkbox("Show only explored steps", value=False)

    with col3:
        max_rows = st.number_input("Rows to show", min_value=10, max_value=5000, value=200, step=10)

    filtered_df = episode_steps_df[episode_steps_df["episode"] == selected_episode].copy()

    if show_only_explored:
        filtered_df = filtered_df[filtered_df["explored"] == True]

    st.subheader(f"Episode {selected_episode} - Step Details")
    st.dataframe(
        filtered_df.head(max_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Episode Summary")
    summary_df = (
        episode_steps_df.groupby("episode")
        .agg(
            steps=("step", "max"),
            total_reward=("reward", "sum"),
            avg_reward=("reward", "mean"),
            total_water_mm=("action_mm", "sum"),
            explored_steps=("explored", "sum"),
            avg_next_soil=("next_soil", "mean"),
        )
        .reset_index()
    )
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

else:
    st.info("No per-episode step logs yet. Train the agent with logging enabled.")


# -----------------------------
# Model controls
# -----------------------------
st.markdown("---")
st.subheader("Model Controls")

col_a, col_b, col_c = st.columns(3)

with col_a:
    if save_button:
        try:
            os.makedirs(ARTIFACT_DIR, exist_ok=True)
            agent.save(SAVE_PATH)
            st.success(f"Saved to {SAVE_PATH}")
        except Exception as e:
            st.error(f"Save failed: {e}")

with col_b:
    if st.button("Show Current Policy"):
        try:
            current_state = env._get_state() if env.index < env.max_steps else env.reset()
            summary = agent.policy_summary(current_state)
            st.json(summary)
        except Exception as e:
            st.error(f"Could not compute policy summary: {e}")

with col_c:
    if st.button("Run Greedy Evaluation"):
        try:
            eval_env = IrrigationEnv(
                data=data,
                action_space=actions,
                target_soil_range=(target_low, target_high),
                eto_noise_std=eto_noise_std,
                rain_noise_std=rain_noise_std,
                reward_scale=reward_scale,
                seed=7,
            )
            s = eval_env.reset()
            total_reward = 0.0
            water_used = 0.0
            eval_history = []

            done = False
            while not done:
                action_idx = agent.predict_action(s)
                ns, r, done, info = eval_env.step(action_idx)
                total_reward += r
                water_used += info["action_mm"]
                eval_history.append(info)
                s = ns

            st.success(f"Greedy eval done | reward: {total_reward:.2f} | water used: {water_used:.2f}")
            st.dataframe(pd.DataFrame(eval_history).tail(20), use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Evaluation failed: {e}")


# -----------------------------
# Smart Predictive Irrigation Planner
# -----------------------------
st.markdown("---")
st.header("🚀 Smart Predictive Irrigation Planner")

if os.path.exists(SAVE_PATH):
    st.success("Saved model detected. You can generate intelligent irrigation plans.")

    with st.expander("Fill Field Conditions for Prediction", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            weather_type = st.selectbox(
                "Weather Type",
                ["Sunny", "Cloudy", "Rainy", "Windy", "Humid", "Dry Heat"]
            )
            last_rain_days = st.selectbox(
                "Last Rain Occurred (days ago)",
                [1, 2, 3, 5, 10, 15, 20, 30]
            )

        with col2:
            temp_user = st.number_input("Temperature (°C)", value=30.0)
            humidity_user = st.number_input("Humidity (%)", value=40.0)
            wind_user = st.number_input("Wind Speed", value=4.0)

        with col3:
            last_watered = st.selectbox(
                "Last Watered Crops",
                ["Today", "Yesterday", "3 days ago", "1 week ago", "2 weeks ago", "1 month ago"]
            )
            field_area = st.number_input("Field Area (hectare)", value=1.0, min_value=0.01)

        generate_plan_btn = st.button("Generate Smart 10-Day Plan")

    if generate_plan_btn:
        try:
            saved_agent = load_saved_agent()

            weather_map = {
                "Sunny": (0.10, 6.0, 0.08),
                "Cloudy": (0.30, 4.0, 0.05),
                "Rainy": (0.85, 2.0, 0.20),
                "Windy": (0.20, 5.0, 0.10),
                "Humid": (0.40, 3.0, 0.06),
                "Dry Heat": (0.05, 7.0, 0.12),
            }
            rain_prob, eto_user, par_user = weather_map[weather_type]

            irrigation_map = {
                "Today": (0, 10),
                "Yesterday": (1, 8),
                "3 days ago": (3, 5),
                "1 week ago": (7, 2),
                "2 weeks ago": (14, 0),
                "1 month ago": (30, 0),
            }
            days_since_last, last_irrigation_mm = irrigation_map[last_watered]

            rain_decay_factor = max(0.0, 1.0 - (last_rain_days / 40.0))
            rain_prob = clamp(rain_prob * rain_decay_factor, 0.0, 1.0)

            base_soil = 50.0
            starting_soil = estimate_starting_soil(
                current_soil=base_soil,
                last_irrigation_mm=last_irrigation_mm,
                days_since_last=days_since_last,
                eto=eto_user,
            )

            forecast_df = build_10_day_forecast_from_user(
                starting_soil=starting_soil,
                temp=temp_user,
                humidity=humidity_user,
                wind=wind_user,
                par=par_user,
                eto=eto_user,
                rain_prob=rain_prob,
                days=10,
            )

            schedule_df, planner_history, total_mm, total_liters, day1_policy = run_10_day_schedule(
                agent=saved_agent,
                forecast_df=forecast_df,
                action_space=actions,
                target_low=target_low,
                target_high=target_high,
                starting_soil=starting_soil,
                field_area_ha=field_area,
                weather_type=weather_type,
                eto_noise_std=eto_noise_std,
                rain_noise_std=rain_noise_std,
                reward_scale=reward_scale,
            )

            baseline_mm = 10.0 * len(schedule_df)
            baseline_liters = baseline_mm * field_area * 10000.0

            saving = max(0.0, baseline_liters - total_liters)
            saving_pct = 0.0 if baseline_liters <= 0 else (saving / baseline_liters) * 100.0

            st.success("✅ Smart irrigation plan generated!")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Water Used", f"{total_liters:,.0f} L")
            c2.metric("Water Saved", f"{saving:,.0f} L")
            c3.metric("Saving %", f"{saving_pct:.2f}%")
            c4.metric("Day 1 Suggested", f"{schedule_df.iloc[0]['action_mm']:.2f} mm")

            st.subheader("📅 10-Day Smart Schedule")
            st.dataframe(schedule_df, use_container_width=True, hide_index=True)

            st.subheader("Day-1 Policy Snapshot")
            st.json(day1_policy)

            fig1, ax1 = plt.subplots(figsize=(10, 4))
            ax1.plot(schedule_df["day"], schedule_df["action_mm"], marker="o")
            ax1.set_title("Daily Irrigation Plan")
            ax1.set_xlabel("Day")
            ax1.set_ylabel("Water (mm)")
            ax1.grid(True, alpha=0.3)
            st.pyplot(fig1)
            plt.close(fig1)

            fig2, ax2 = plt.subplots(figsize=(10, 4))
            ax2.plot(schedule_df["day"], schedule_df["soil_before"], marker="o", label="Soil Before")
            ax2.plot(schedule_df["day"], schedule_df["soil_after"], marker="o", label="Soil After")
            ax2.set_title("10-Day Soil Trend")
            ax2.set_xlabel("Day")
            ax2.set_ylabel("Soil")
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            st.pyplot(fig2)
            plt.close(fig2)

            fig3, ax3 = plt.subplots(figsize=(10, 4))
            ax3.bar(schedule_df["day"].astype(str), schedule_df["water_liters"])
            ax3.set_title("Daily Water Use (Liters)")
            ax3.set_xlabel("Day")
            ax3.set_ylabel("Liters")
            ax3.grid(True, axis="y", alpha=0.3)
            st.pyplot(fig3)
            plt.close(fig3)

        except Exception as e:
            st.error(f"Planner failed: {e}")
else:
    st.warning("⚠️ Please train and save the agent first to use the predictive planner.")