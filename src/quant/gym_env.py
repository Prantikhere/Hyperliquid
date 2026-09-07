"""
Gymnasium adapter around the existing hand-rolled TradingEnv (src/quant/env.py).

TradingEnv is TensorTrade-inspired but predates gym/gymnasium wiring -- this module
is the thin bridge so it can be trained with stable-baselines3 (SB3), without pulling
in the actual `tensortrade` package (unmaintained, pinned to pre-gymnasium `gym` and
numpy<2 -- incompatible with this venv's Python 3.14 / numpy2 stack).

Observation: [normalized_price_change_from_window_start, position] -- position in
{-1, 0, 1} lets the policy condition on whether it's already in a trade.
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
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        self._ref_price = float(prices[0])

    def _obs(self, raw_price):
        pct_change = (raw_price - self._ref_price) / self._ref_price
        return np.array([pct_change, float(self.env.position)], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        raw = self.env.reset()
        self._ref_price = float(self.env.prices[0])
        return self._obs(raw), {}

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(int(action))
        obs = self._obs(raw_obs)
        # gymnasium splits `done` into terminated (natural end) vs truncated (time limit) --
        # TradingEnv only has one notion of "ran out of data", so treat it as terminated.
        return obs, float(reward), bool(done), False, info
