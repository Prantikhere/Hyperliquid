import numpy as np
import pandas as pd
from src.quant.reward_scheme import SimpleProfit, RiskAdjustedReturns

class TradingEnv:
    """
    Modular TensorTrade-inspired Trading Environment.
    Wraps asset price streams, action execution, and reward schemes.

    Reward: per-step risk-adjusted return (Sharpe of the last N returns),
    NOT cumulative Sortino over the full episode. This gives PPO a meaningful
    gradient signal at every step.
    """
    REWARD_WINDOW = 20  # lookback for rolling Sharpe reward

    def __init__(self, prices: list, initial_balance: float = 1.0, fee: float = 0.0005, reward_scheme = None):
        self.prices = np.array(prices)
        self.initial_balance = initial_balance
        self.fee = fee
        self.reward_scheme = reward_scheme or SimpleProfit()
        self.reset()

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.position = 0  # +1: Long, -1: Short, 0: Flat
        self.entry_price = 0.0
        self.net_worths = [self.initial_balance]
        self.returns = []
        self.done = False
        return self._get_observation()

    def _get_observation(self):
        if self.current_step < len(self.prices):
            return self.prices[self.current_step]
        return self.prices[-1]

    def step(self, action: int):
        """
        Actions:
        0: HOLD / Keep current position (or do nothing if flat)
        1: BUY / Go Long (close short if exists, enter long)
        2: SELL / Go Short (close long if exists, enter short)
        """
        if self.current_step >= len(self.prices) - 1:
            self.done = True
            return self._get_observation(), 0.0, self.done, {}

        price = self.prices[self.current_step]
        next_price = self.prices[self.current_step + 1]

        # Execute order & calculate fee drag
        if action == 1 and self.position <= 0:  # BUY / Long
            if self.position == -1:  # Close Short
                ret = (self.entry_price - price) / self.entry_price
                self.balance *= (1.0 + ret - self.fee)
            self.position = 1
            self.entry_price = price
            self.balance *= (1.0 - self.fee)
        elif action == 2 and self.position >= 0:  # SELL / Short
            if self.position == 1:  # Close Long
                ret = (price - self.entry_price) / self.entry_price
                self.balance *= (1.0 + ret - self.fee)
            self.position = -1
            self.entry_price = price
            self.balance *= (1.0 - self.fee)
        elif action == 0 and self.position != 0:  # HOLD but in position
            pass

        # Calculate current net worth at the next step's price
        if self.position == 1:
            current_net_worth = self.balance * (next_price / self.entry_price)
        elif self.position == -1:
            # Short P&L: profit when price falls. Floor at 0.01 to avoid negative net_worth
            # (margin call scenario). In production, margin mode would handle this.
            pnl_mult = 2.0 - (next_price / self.entry_price)
            current_net_worth = self.balance * max(pnl_mult, 0.01)
        else:
            current_net_worth = self.balance

        # Track returns and net worth
        prev_net_worth = self.net_worths[-1]
        step_return = (current_net_worth - prev_net_worth) / prev_net_worth
        self.returns.append(step_return)
        self.net_worths.append(current_net_worth)

        # Advance environment
        self.current_step += 1
        if self.current_step >= len(self.prices) - 1:
            self.done = True

        # Per-step reward: rolling Sharpe of last N returns (NOT cumulative over episode).
        # This gives PPO a meaningful gradient signal at every step.
        reward = self._per_step_reward()

        return self._get_observation(), reward, self.done, {
            "net_worth": current_net_worth,
            "position": self.position,
            "balance": self.balance
        }

    def _per_step_reward(self):
        """Compute per-step reward as rolling risk-adjusted return."""
        n = len(self.returns)
        if n < 3:
            # Not enough data for meaningful reward; use raw step return
            return self.returns[-1] if self.returns else 0.0

        window = self.returns[-self.REWARD_WINDOW:]
        ret_arr = np.array(window)
        mean_r = ret_arr.mean()
        std_r = ret_arr.std()

        if std_r < 1e-10:
            return mean_r  # no volatility -> reward = raw mean return

        # Rolling Sharpe (annualized for hourly data, but we keep it raw for scale)
        sharpe = mean_r / std_r

        # Clip to [-5, 5] to prevent extreme rewards from dominating the gradient
        return float(np.clip(sharpe, -5.0, 5.0))
