import numpy as np
import pandas as pd
class IrrigationEnv:
    def __init__(
        self,
        data,
        action_space=None,
        target_soil_range=None,
        soil_min=None,
        soil_max=None,
        eto_noise_std=0.8,
        rain_noise_std=0.5,
        evaporation_factor=0.15,
        irrigation_efficiency=0.9,
        drainage_factor=0.05,
        crop_sensitivity=1.0,
        reward_scale=1.0,
        seed=42,
    ):
        """
        RL environment for smart irrigation.

        Expected columns in data:
        - soil
        - temp
        - humidity
        - wind
        - par
        - eto

        Optional:
        - etr
        - rain
        - rain_prob
        - water_demand

        If target_soil_range is None, it is inferred from the soil distribution
        so the environment works even when soil values are on a large raw scale.
        """
        self.data = data.reset_index(drop=True).copy()
        self.index = 0
        self.max_steps = len(self.data)
        self.action_space = action_space if action_space is not None else [0, 5, 10, 15, 20]
        self.n_actions = len(self.action_space)
        self.eto_noise_std = float(eto_noise_std)
        self.rain_noise_std = float(rain_noise_std)
        self.evaporation_factor = float(evaporation_factor)
        self.irrigation_efficiency = float(irrigation_efficiency)
        self.drainage_factor = float(drainage_factor)
        self.crop_sensitivity = float(crop_sensitivity)
        self.reward_scale = float(reward_scale)
        self.rng = np.random.default_rng(seed)
        if soil_min is None:
            soil_min = float(self.data["soil"].quantile(0.01)) if "soil" in self.data.columns else 0.0
        if soil_max is None:
            soil_max = float(self.data["soil"].quantile(0.99)) if "soil" in self.data.columns else 100.0
        if soil_max <= soil_min:
            soil_max = soil_min + 1.0
        self.soil_min = float(soil_min)
        self.soil_max = float(soil_max)
        if target_soil_range is None:
            if "soil" in self.data.columns:
                t_low = float(self.data["soil"].quantile(0.35))
                t_high = float(self.data["soil"].quantile(0.65))
            else:
                t_low, t_high = 35.0, 70.0

            self.target_low = min(t_low, t_high)
            self.target_high = max(t_low, t_high)
        else:
            self.target_low, self.target_high = target_soil_range

            # If target band looks percentage-like but soil scale is huge,
            # auto-switch to quantile-based band
            if self.soil_max > 20 * max(self.target_high, 1.0):
                if "soil" in self.data.columns:
                    t_low = float(self.data["soil"].quantile(0.35))
                    t_high = float(self.data["soil"].quantile(0.65))
                    self.target_low = min(t_low, t_high)
                    self.target_high = max(t_low, t_high)

        self.current_soil = None
        self.history = []

    # -------------------------
    # Basic helpers
    # -------------------------
    def _safe_get(self, row, key, default=0.0):
        val = row.get(key, default)
        if pd.isna(val):
            return default
        return float(val)

    def _normalize_soil(self, soil_value):
        return float(
            np.clip(
                (soil_value - self.soil_min) / (self.soil_max - self.soil_min + 1e-8),
                0.0,
                1.0,
            )
        )

    def _target_band_normalized(self):
        low_n = self._normalize_soil(self.target_low)
        high_n = self._normalize_soil(self.target_high)
        return min(low_n, high_n), max(low_n, high_n)

    def _moisture_gap(self, soil_value):
        """
        Signed gap from target band:
        - positive if below target_low
        - negative if above target_high
        - 0 if inside band
        """
        if soil_value < self.target_low:
            return self.target_low - soil_value
        elif soil_value > self.target_high:
            return self.target_high - soil_value
        return 0.0

    def _distance_to_band(self, soil_value):
        """
        Absolute distance from the nearest target boundary.
        """
        if soil_value < self.target_low:
            return self.target_low - soil_value
        elif soil_value > self.target_high:
            return soil_value - self.target_high
        return 0.0

    def _get_demand(self, row):
        """
        Prefer ETo, fallback to ETr, then water_demand.
        """
        val = row.get("eto", None)
        if val is not None and not pd.isna(val):
            return float(val)

        val = row.get("etr", None)
        if val is not None and not pd.isna(val):
            return float(val)

        val = row.get("water_demand", None)
        if val is not None and not pd.isna(val):
            return float(val)

        return 0.0

    def _get_rain(self, row):
        """
        Use rain / rain_prob if present, otherwise 0.
        """
        val = row.get("rain", None)
        if val is not None and not pd.isna(val):
            return float(val)

        val = row.get("rain_prob", None)
        if val is not None and not pd.isna(val):
            return float(val) * 2.0

        return 0.0

    # -------------------------
    # RL API
    # -------------------------
    def reset(self):
        self.index = 0
        self.history = []

        first_row = self.data.iloc[self.index]
        self.current_soil = self._safe_get(first_row, "soil", default=self.target_low)

        return self._get_state()

    def _get_state(self):
        row = self.data.iloc[self.index]

        soil = self.current_soil
        temp = self._safe_get(row, "temp", 0.0)
        humidity = self._safe_get(row, "humidity", 0.0)
        wind = self._safe_get(row, "wind", 0.0)
        par = self._safe_get(row, "par", 0.0)
        eto = self._get_demand(row)
        rain = self._get_rain(row)

        soil_norm = self._normalize_soil(soil)
        moisture_gap = self._moisture_gap(soil)
        demand_pressure = max(0.0, eto - rain)
        climate_stress = max(0.0, temp - humidity * 0.1)

        return np.array(
            [
                soil,
                temp,
                humidity,
                wind,
                par,
                eto,
                rain,
                soil_norm,
                moisture_gap,
                demand_pressure,
                climate_stress,
            ],
            dtype=np.float32,
        )

    def _resolve_action(self, action):
        """
        Accept either action index or actual irrigation amount.
        """
        if isinstance(action, (int, np.integer)) and 0 <= int(action) < self.n_actions:
            action_idx = int(action)
            action_mm = float(self.action_space[action_idx])
        else:
            action_mm = float(action)
            if action_mm in self.action_space:
                action_idx = self.action_space.index(action_mm)
            else:
                action_idx = int(np.argmin([abs(a - action_mm) for a in self.action_space]))
                action_mm = float(self.action_space[action_idx])

        return action_idx, action_mm

    def _apply_dynamics(self, action_mm, row):
        """
        next_soil = current_soil + irrigation + rain - losses - drainage
        """
        eto = self._get_demand(row)
        rain = self._get_rain(row)

        # uncertainty
        eto_noisy = max(0.0, eto + self.rng.normal(0, self.eto_noise_std))
        rain_noisy = max(0.0, rain + self.rng.normal(0, self.rain_noise_std))

        # losses
        natural_loss = eto_noisy * self.evaporation_factor

        # effective irrigation
        effective_irrigation = action_mm * self.irrigation_efficiency

        # drainage if soil goes above target band
        projected_soil = self.current_soil + effective_irrigation + rain_noisy - natural_loss
        drainage = 0.0
        if projected_soil > self.target_high:
            drainage = (projected_soil - self.target_high) * self.drainage_factor

        next_soil = self.current_soil + effective_irrigation + rain_noisy - natural_loss - drainage
        next_soil = float(np.clip(next_soil, self.soil_min, self.soil_max))

        return next_soil, eto_noisy, rain_noisy, natural_loss, effective_irrigation, drainage

    def _compute_reward(
        self,
        action_mm,
        next_soil,
        eto_noisy,
        rain_noisy,
        natural_loss,
        effective_irrigation,
        drainage,
    ):
        """
        Reward logic:
        - strong positive reward for staying in target band
        - strong penalty for not irrigating when soil is too low
        - mild penalty for excess water use
        - mild penalty for irrigating when rain is already helping
        """

        # Reward for soil condition
        if self.target_low <= next_soil <= self.target_high:
            soil_reward = 20.0
        else:
            dist = self._distance_to_band(next_soil)
            range_span = max(1.0, self.target_high - self.target_low)
            soil_reward = -self.crop_sensitivity * (dist / range_span) * 10.0

        demand_balance = abs((effective_irrigation + rain_noisy) - natural_loss)

        # Under-irrigation penalty
        if action_mm == 0 and next_soil < self.target_low:
            under_irrigation_penalty = -8.0
        elif effective_irrigation < natural_loss * 0.7:
            under_irrigation_penalty = -3.0
        else:
            under_irrigation_penalty = -0.05 * demand_balance

        # Rain + irrigation overlap penalty
        if effective_irrigation > 0 and rain_noisy > 0:
            rain_overirrigation_penalty = -2.0
        else:
            rain_overirrigation_penalty = 0.0

        # Overuse penalties
        over_irrigation_penalty = -0.02 * action_mm
        drainage_penalty = -0.2 * drainage

        # Bonus if soil stays stable in desired band
        stability_bonus = 5.0 if self.target_low <= next_soil <= self.target_high else 0.0

        reward = (
            soil_reward
            + under_irrigation_penalty
            + rain_overirrigation_penalty
            + over_irrigation_penalty
            + drainage_penalty
            + stability_bonus
        )

        # Discourage always choosing zero irrigation
        if action_mm == 0:
            reward -= 0.5

        return float(reward * self.reward_scale)

    def step(self, action):
        if self.index >= self.max_steps:
            raise RuntimeError("Episode already finished. Call reset() first.")

        row = self.data.iloc[self.index]
        prev_soil = float(self.current_soil)

        action_idx, action_mm = self._resolve_action(action)

        next_soil, eto_noisy, rain_noisy, natural_loss, effective_irrigation, drainage = self._apply_dynamics(
            action_mm, row
        )

        reward = self._compute_reward(
            action_mm=action_mm,
            next_soil=next_soil,
            eto_noisy=eto_noisy,
            rain_noisy=rain_noisy,
            natural_loss=natural_loss,
            effective_irrigation=effective_irrigation,
            drainage=drainage,
        )

        self.history.append(
            {
                "step": self.index,
                "prev_soil": prev_soil,
                "action_idx": action_idx,
                "action_mm": action_mm,
                "eto": self._get_demand(row),
                "eto_noisy": eto_noisy,
                "rain": self._get_rain(row),
                "rain_noisy": rain_noisy,
                "natural_loss": natural_loss,
                "effective_irrigation": effective_irrigation,
                "drainage": drainage,
                "next_soil": next_soil,
                "reward": reward,
                "soil_norm": self._normalize_soil(next_soil),
                "moisture_gap": self._moisture_gap(next_soil),
            }
        )

        self.current_soil = next_soil
        self.index += 1
        done = self.index >= self.max_steps
        next_state = self._get_state() if not done else None

        info = {
            "action_idx": action_idx,
            "action_mm": action_mm,
            "prev_soil": prev_soil,
            "next_soil": next_soil,
            "eto": self._get_demand(row),
            "eto_noisy": eto_noisy,
            "rain": self._get_rain(row),
            "rain_noisy": rain_noisy,
            "natural_loss": natural_loss,
            "effective_irrigation": effective_irrigation,
            "drainage": drainage,
            "soil_norm": self._normalize_soil(next_soil),
            "moisture_gap": self._moisture_gap(next_soil),
        }

        return next_state, reward, done, info

    # -------------------------
    # Extra utilities
    # -------------------------
    def get_history_df(self):
        return pd.DataFrame(self.history)

    def get_current_info(self):
        if self.index >= self.max_steps:
            return {}

        row = self.data.iloc[self.index]
        return {
            "step": self.index,
            "soil": float(self.current_soil),
            "eto": self._get_demand(row),
            "rain": self._get_rain(row),
        }