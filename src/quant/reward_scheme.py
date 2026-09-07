import numpy as np
import pandas as pd

class RewardScheme:
    """Base class for all reward schemes, matching the modular design of TensorTrade."""
    def get_reward(self, returns: pd.Series) -> float:
        raise NotImplementedError()

class SimpleProfit(RewardScheme):
    """
    Simple Profit/Loss reward scheme.
    Returns the sum of returns, representing the cumulative net profit.
    """
    def get_reward(self, returns: pd.Series) -> float:
        if returns.empty:
            return 0.0
        return float(returns.sum())

class RiskAdjustedReturns(RewardScheme):
    """
    Risk-Adjusted Returns reward scheme.
    Supports Sharpe and Sortino ratio calculations over a return series.
    """
    def __init__(self, return_algorithm: str = 'sortino', risk_free_rate: float = 0.0, target_returns: float = 0.0):
        self.return_algorithm = return_algorithm.lower()
        self.risk_free_rate = risk_free_rate
        self.target_returns = target_returns

    def get_reward(self, returns: pd.Series) -> float:
        if returns.empty or len(returns) < 2:
            return 0.0

        if self.return_algorithm == 'sharpe':
            return self._sharpe_ratio(returns)
        elif self.return_algorithm == 'sortino':
            return self._sortino_ratio(returns)
        else:
            raise ValueError(f"Unknown return algorithm: {self.return_algorithm}")

    def _sharpe_ratio(self, returns: pd.Series) -> float:
        mean_return = returns.mean()
        std_return = returns.std()
        if std_return == 0 or np.isnan(std_return):
            return 0.0
        # Annualized Sharpe (assuming hourly data, 8760 hours in a year)
        return float((mean_return - self.risk_free_rate) / std_return * np.sqrt(8760))

    def _sortino_ratio(self, returns: pd.Series) -> float:
        mean_return = returns.mean()
        downside_returns = returns[returns < self.target_returns]
        
        if downside_returns.empty:
            return 0.0
            
        # Downside deviation uses mean squared negative returns
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
        if downside_deviation == 0 or np.isnan(downside_deviation):
            return 0.0
            
        # Annualized Sortino (assuming hourly data)
        return float((mean_return - self.target_returns) / downside_deviation * np.sqrt(8760))
