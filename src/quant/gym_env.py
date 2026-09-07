"""
Gymnasium adapter around the existing hand-rolled TradingEnv (src/quant/env.py).

TradingEnv is TensorTrade-inspired but predates gym/gymnasium wiring -- this module
is the thin bridge so it can be trained with stable-baselines3 (SB3), without pulling
in the actual `tensortrade` package (unmaintained, pinned to pre-gymnasium `gym` and
numpy<2 -- incompatible with this venv's Python 3.14 / numpy2 stack).

Observation (8 features):
  0: normalized_price_change_from_window_start
  1: position (-1, 0, +1)
  2: RSI-14 (0-100, normalized to 0-1)
  3: short_term_momentum (5-bar return)
  4: medium_term_momentum (20-bar return)
  5: volatility_ratio (recent vol / full vol)
  6: price_distance_from_SMA20
  7: z_score (20-bar)
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from src.quant.env import TradingEnv


class TradingGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, prices, fee=0.0005, reward_scheme=None):
        super().__init__()
        self.env = TradingEnv(prices, fee=fee, reward_scheme=reward_scheme)
        self.action_space = spaces.Discrete(3)  # 0=HOLD, 1=BUY/LONG, 2=SELL/SHORT
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self._prices = np.array(prices, dtype=np.float64)
        self._ref_price = float(prices[0])
        # Pre-compute indicators for the full price series (fast, no per-step overhead)
        self._rsi = self._compute_rsi(self._prices, 14)
        self._sma20 = self._rolling_mean(self._prices, 20)
        self._std20 = self._rolling_std(self._prices, 20)

    @staticmethod
    def _rolling_mean(arr, window):
        out = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            out[i] = arr[i - window + 1:i + 1].mean()
        return out

    @staticmethod
    def _rolling_std(arr, window):
        out = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            out[i] = arr[i - window + 1:i + 1].std()
        return out

    @staticmethod
    def _compute_rsi(prices, period=14):
        """Wilder's RSI over the full price array."""
        deltas = np.diff(prices, prepend=prices[0])
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.full_like(prices, np.nan)
        avg_loss = np.full_like(prices, np.nan)
        rsi = np.full_like(prices, 50.0)  # default neutral
        if len(gains) < period + 1:
            return rsi
        avg_gain[period] = gains[1:period + 1].mean()
        avg_loss[period] = losses[1:period + 1].mean()
        for i in range(period + 1, len(prices)):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i]) / period
        for i in range(period, len(prices)):
            if avg_loss[i] == 0:
                rsi[i] = 100.0
            else:
                rs = avg_gain[i] / avg_loss[i]
                rsi[i] = 100.0 - (100.0 / (1.0 + rs))
        return rsi

    def _obs(self, step_idx):
        """Build 8-feature observation vector from pre-computed indicators."""
        price = self._prices[step_idx]
        pct_change = (price - self._ref_price) / self._ref_price if self._ref_price > 0 else 0.0

        # RSI normalized to 0-1
        rsi = self._rsi[step_idx] / 100.0 if not np.isnan(self._rsi[step_idx]) else 0.5

        # Short-term momentum (5-bar return)
        mom5 = 0.0
        if step_idx >= 5:
            mom5 = (price - self._prices[step_idx - 5]) / self._prices[step_idx - 5]

        # Medium-term momentum (20-bar return)
        mom20 = 0.0
        if step_idx >= 20:
            mom20 = (price - self._prices[step_idx - 20]) / self._prices[step_idx - 20]

        # Volatility ratio (recent 5-bar std / full 20-bar std)
        vol_ratio = 1.0
        if step_idx >= 20:
            recent_std = self._prices[step_idx - 4:step_idx + 1].std()
            full_std = self._std20[step_idx] if not np.isnan(self._std20[step_idx]) else 1e-9
            vol_ratio = recent_std / full_std if full_std > 1e-9 else 1.0

        # Distance from SMA20 (normalized)
        sma = self._sma20[step_idx] if not np.isnan(self._sma20[step_idx]) else price
        dist_sma = (price - sma) / sma if sma > 0 else 0.0

        # Z-score (20-bar)
        std20 = self._std20[step_idx] if not np.isnan(self._std20[step_idx]) else 1e-9
        zscore = (price - sma) / std20 if std20 > 1e-9 else 0.0
        zscore = np.clip(zscore, -3.0, 3.0)  # clip extreme z-scores

        return np.array([
            pct_change,
            float(self.env.position),
            rsi,
            mom5,
            mom20,
            vol_ratio,
            dist_sma,
            zscore,
        ], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        raw = self.env.reset()
        self._ref_price = float(self._prices[0])
        return self._obs(0), {}

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(int(action))
        obs = self._obs(self.env.current_step)
        return obs, float(reward), bool(done), False, info
