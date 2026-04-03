import numpy as np
import random
import pickle
from collections import defaultdict
from typing import List, Optional, Tuple, Union, Dict, Any


class QLearningAgent:
    def __init__(
        self,
        action_space: List[float],
        lr: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.10,
        epsilon_decay: float = 0.999,
        state_decimals: int = 1,
        use_discretization: bool = False,
        n_bins: int = 10,
        seed: Optional[int] = 42,
        optimistic_init: float = 0.0,
    ):
        """
        Q-learning agent for irrigation scheduling.

        Parameters
        ----------
        action_space : list
            Example: [0, 5, 10, 15, 20]
        lr : float
            Learning rate.
        gamma : float
            Discount factor.
        epsilon : float
            Exploration rate.
        epsilon_min : float
            Minimum exploration value.
        epsilon_decay : float
            Multiplicative epsilon decay after each episode.
        state_decimals : int
            Rounding precision for tabular state hashing.
        use_discretization : bool
            If True, continuous states are binned before hashing.
        n_bins : int
            Number of bins per state feature when discretization is enabled.
        seed : int or None
            Random seed for reproducibility.
        optimistic_init : float
            Initial Q-value for unseen states. Can help exploration.
        """
        self.action_space = list(action_space)
        self.n_actions = len(self.action_space)

        self.lr = float(lr)
        self.gamma = float(gamma)

        self.epsilon = float(epsilon)
        self.epsilon_min = float(epsilon_min)
        self.epsilon_decay = float(epsilon_decay)

        self.state_decimals = int(state_decimals)
        self.use_discretization = bool(use_discretization)
        self.n_bins = int(n_bins)
        self.optimistic_init = float(optimistic_init)

        self.q_table = defaultdict(
            lambda: np.full(self.n_actions, self.optimistic_init, dtype=np.float32)
        )

        self.training_log: List[Dict[str, Any]] = []
        self.episode_log: List[Dict[str, Any]] = []

        self.feature_bounds = None
        self.rng = np.random.default_rng(seed)
        random.seed(seed)
        np.random.seed(seed if seed is not None else None)

    def set_feature_bounds(self, bounds: List[Tuple[float, float]]):
        self.feature_bounds = bounds

    def _discretize_value(self, value: float, low: float, high: float) -> int:
        if high <= low:
            return 0
        value = float(np.clip(value, low, high))
        bin_width = (high - low) / self.n_bins
        if bin_width == 0:
            return 0
        bin_idx = int((value - low) / bin_width)
        return min(bin_idx, self.n_bins - 1)

    def get_state_key(self, state: Union[np.ndarray, List[float], Tuple[float, ...]]):
        if state is None:
            raise ValueError("State cannot be None.")

        state = np.asarray(state, dtype=np.float32)

        if self.use_discretization:
            if self.feature_bounds is None:
                raise ValueError("feature_bounds must be set when use_discretization=True")
            if len(self.feature_bounds) != len(state):
                raise ValueError(
                    f"feature_bounds length ({len(self.feature_bounds)}) does not match "
                    f"state length ({len(state)})."
                )

            key = []
            for i, val in enumerate(state):
                low, high = self.feature_bounds[i]
                key.append(self._discretize_value(val, low, high))
            return tuple(key)

        return tuple(np.round(state, self.state_decimals))

    def state_seen(self, state) -> bool:
        state_key = self.get_state_key(state)
        return state_key in self.q_table

    def choose_action(self, state, training: bool = True, return_info: bool = False):
        """
        Epsilon-greedy action selection.
        Returns action index. Optionally returns action info.
        """
        state_key = self.get_state_key(state)
        seen = state_key in self.q_table

        explored = False
        if training and random.random() < self.epsilon:
            action_idx = random.randrange(self.n_actions)
            explored = True
        else:
            q_values = self.q_table[state_key]
            action_idx = int(np.argmax(q_values))

        if return_info:
            q_values = self.q_table[state_key].copy()
            return action_idx, {
                "state_key": state_key,
                "explored": explored,
                "epsilon": float(self.epsilon),
                "q_values": q_values,
                "action_mm": self.action_space[action_idx],
                "state_seen": seen,
            }

        return action_idx

    def get_q_values(self, state):
        state_key = self.get_state_key(state)
        return self.q_table[state_key].copy()

    def _heuristic_action_for_unseen_state(self, state) -> int:
        """
        Fallback used in planner / greedy inference when the exact state
        was never seen during training.

        Expected state format from environment:
        [soil, temp, humidity, wind, par, eto, rain, soil_norm,
         moisture_gap, demand_pressure, climate_stress]
        """
        s = np.asarray(state, dtype=np.float32)

        soil = float(s[0]) if len(s) > 0 else 0.0
        eto = float(s[5]) if len(s) > 5 else 0.0
        rain = float(s[6]) if len(s) > 6 else 0.0
        moisture_gap = float(s[8]) if len(s) > 8 else 0.0
        demand_pressure = float(s[9]) if len(s) > 9 else max(0.0, eto - rain)

        # Heuristic target irrigation in mm
        if moisture_gap > 0:
            # soil below target: irrigate based on demand + gap
            desired_mm = demand_pressure + min(5.0, 0.15 * moisture_gap)
        else:
            # soil not below target: less or no irrigation
            desired_mm = max(0.0, demand_pressure - max(0.0, rain))

        # Clamp to available action space
        desired_mm = max(0.0, desired_mm)

        # Pick nearest valid action
        idx = int(np.argmin([abs(a - desired_mm) for a in self.action_space]))
        return idx

    def predict_action(self, state):
        """
        Greedy action selection only.
        For unseen states, use a heuristic fallback instead of always returning action 0.
        """
        state_key = self.get_state_key(state)

        if state_key not in self.q_table:
            return self._heuristic_action_for_unseen_state(state)

        q_values = self.q_table[state_key]
        if np.allclose(q_values, q_values[0]):
            # all equal => still no learned preference, use heuristic
            return self._heuristic_action_for_unseen_state(state)

        return int(np.argmax(q_values))

    def update(self, state, action_idx, reward, next_state, done: bool = False):
        """
        Q-learning update rule:
        Q(s,a) <- Q(s,a) + alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
        """
        state_key = self.get_state_key(state)
        current_q = self.q_table[state_key][action_idx]

        if done or next_state is None:
            target = reward
        else:
            next_key = self.get_state_key(next_state)
            next_max = np.max(self.q_table[next_key])
            target = reward + self.gamma * next_max

        new_q = current_q + self.lr * (target - current_q)
        self.q_table[state_key][action_idx] = new_q
        return float(new_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return float(self.epsilon)

    def record_step(
        self,
        episode: int,
        step: int,
        state,
        action_idx: int,
        reward: float,
        next_state,
        done: bool,
        info: Optional[Dict[str, Any]] = None,
    ):
        entry = {
            "episode": int(episode),
            "step": int(step),
            "state_key": self.get_state_key(state),
            "action_idx": int(action_idx),
            "action_mm": float(self.action_space[action_idx]),
            "reward": float(reward),
            "done": bool(done),
            "epsilon": float(self.epsilon),
            "state_seen": self.state_seen(state),
        }

        if info is not None:
            entry.update(info)

        self.training_log.append(entry)

    def start_episode(self, episode: int):
        self._episode_reward = 0.0
        self._episode_steps = 0
        self._current_episode = int(episode)

    def end_episode(self):
        summary = {
            "episode": getattr(self, "_current_episode", None),
            "total_reward": float(getattr(self, "_episode_reward", 0.0)),
            "steps": int(getattr(self, "_episode_steps", 0)),
            "epsilon": float(self.epsilon),
        }
        self.episode_log.append(summary)
        return summary

    def track_episode_step(self, reward: float):
        self._episode_reward = getattr(self, "_episode_reward", 0.0) + float(reward)
        self._episode_steps = getattr(self, "_episode_steps", 0) + 1

    def save(self, path: str = "q_table.pkl"):
        payload = {
            "q_table": dict(self.q_table),
            "action_space": self.action_space,
            "lr": self.lr,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "state_decimals": self.state_decimals,
            "use_discretization": self.use_discretization,
            "n_bins": self.n_bins,
            "feature_bounds": self.feature_bounds,
            "training_log": self.training_log,
            "episode_log": self.episode_log,
            "optimistic_init": self.optimistic_init,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str = "q_table.pkl"):
        with open(path, "rb") as f:
            payload = pickle.load(f)

        agent = cls(
            action_space=payload["action_space"],
            lr=payload["lr"],
            gamma=payload["gamma"],
            epsilon=payload["epsilon"],
            epsilon_min=payload["epsilon_min"],
            epsilon_decay=payload["epsilon_decay"],
            state_decimals=payload["state_decimals"],
            use_discretization=payload["use_discretization"],
            n_bins=payload["n_bins"],
            optimistic_init=payload.get("optimistic_init", 0.0),
        )

        loaded_q = payload["q_table"]
        agent.q_table = defaultdict(
            lambda: np.full(agent.n_actions, agent.optimistic_init, dtype=np.float32)
        )
        agent.q_table.update(loaded_q)

        agent.feature_bounds = payload["feature_bounds"]
        agent.training_log = payload.get("training_log", [])
        agent.episode_log = payload.get("episode_log", [])
        return agent

    def reset_logs(self):
        self.training_log = []
        self.episode_log = []

    def q_table_size(self) -> int:
        return len(self.q_table)

    def best_action_for_state(self, state):
        q_values = self.get_q_values(state)
        idx = int(np.argmax(q_values))
        return self.action_space[idx], float(q_values[idx])

    def policy_summary(self, state):
        state_key = self.get_state_key(state)
        seen = state_key in self.q_table

        if seen:
            q_values = self.q_table[state_key].copy()
            best_idx = int(np.argmax(q_values))
            source = "learned_q_table"
        else:
            best_idx = self._heuristic_action_for_unseen_state(state)
            q_values = np.full(self.n_actions, np.nan, dtype=np.float32)
            source = "heuristic_fallback"

        return {
            "state_seen": seen,
            "decision_source": source,
            "best_action_idx": best_idx,
            "best_action_mm": float(self.action_space[best_idx]),
            "best_q_value": None if not seen else float(q_values[best_idx]),
            "q_values": q_values.tolist() if seen else None,
        }