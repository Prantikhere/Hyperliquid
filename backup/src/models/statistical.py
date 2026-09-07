import numpy as np
from src.utils.logger import log

class StatisticalModels:
    def __init__(self):
        pass

    def calculate_z_score(self, current_price, history):
        """Calculate Z-score for mean reversion."""
        if len(history) < 20:
            return 0
        mean = np.mean(history)
        std = np.std(history)
        if std == 0:
            return 0
        return (current_price - mean) / std

    def check_arbitrage(self, yes_price, no_price, fee=0.02):
        """
        Check for Yes + No < 1 (or > 1 if we can sell short, 
        but Polymarket is mostly directional).
        """
        total = yes_price + no_price
        if total < (1.0 - fee):
            return {"type": "arb_buy_both", "ev": 1.0 - total - fee}
        return None

    def get_signal(self, features):
        """Simple stat-arb signal based on features."""
        # Example: If imbalance is high and relative spread is low, signal buy
        imbalance = features.get("order_imbalance", 0.5)
        if imbalance > 0.8:
            return {"side": "buy", "confidence": 0.6}
        elif imbalance < 0.2:
            return {"side": "sell", "confidence": 0.6}
        return None
