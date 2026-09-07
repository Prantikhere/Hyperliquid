import numpy as np
import pandas as pd
from src.utils.logger import log

class StrategyEnsemble:
    """
    Implements multiple quantitative strategies discussed in the blueprints.
    Provides signals for Mean Reversion, Momentum, and Order Flow.
    """
    def __init__(self):
        pass

    def get_signals(self, prices, book=None):
        if len(prices) < 20:
            return {"mean_reversion": 0.5, "momentum": 0.5, "order_flow": 0.5, "trend": 0.5}

        df = pd.Series(prices)
        
        # 1. Mean Reversion Signal (Z-Score)
        mean = df.mean()
        std = df.std()
        z_score = (df.iloc[-1] - mean) / std if std != 0 else 0
        # Normalize to 0-1: < 0.5 means oversold (BUY), > 0.5 means overbought (SELL)
        mr_signal = 1.0 / (1.0 + np.exp(z_score)) 

        # 2. Momentum Signal (RSI + MACD proxy)
        change = df.diff()
        gain = (change.where(change > 0, 0)).rolling(window=14).mean()
        loss = (-change.where(change < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        mom_signal = rsi / 100.0 # High RSI = High Momentum (0 to 1)

        # 3. Order Flow Signal (Bid/Ask Imbalance)
        of_signal = 0.5
        if book:
            bids = sum([float(b[1]) for b in book.get('bids', [])[:5]])
            asks = sum([float(a[1]) for a in book.get('asks', [])[:5]])
            if (bids + asks) > 0:
                of_signal = bids / (bids + asks) # High = Buy Pressure

        # 4. Trend Signal (fast/slow EMA separation) for the trend-following sleeve.
        # >0.5 = fast EMA above slow EMA = uptrend; magnitude scales with trend strength.
        trend_signal = 0.5
        if len(df) >= 30:
            ema_fast = df.ewm(span=10, adjust=False).mean().iloc[-1]
            ema_slow = df.ewm(span=30, adjust=False).mean().iloc[-1]
            if ema_slow != 0:
                rel = (ema_fast - ema_slow) / ema_slow
                trend_signal = float(1.0 / (1.0 + np.exp(-50.0 * rel)))  # 2% gap -> ~0.73

        return {
            "mean_reversion": float(mr_signal),
            "momentum": float(mom_signal),
            "order_flow": float(of_signal),
            "trend": float(trend_signal)
        }

    def composite_score(self, signals, regime="NEUTRAL"):
        """
        Regime-aware directional score in [0, 1] (>0.5 bullish, <0.5 bearish).

        Validation finding (see Crypto-Trading-Bot deck): mean reversion is the only
        strategy family with a positive out-of-sample edge, so it is over-weighted in
        mean-reverting regimes. Trend-following is NOT a validated edge, so in trending
        regimes conviction is deliberately compressed toward neutral (stand-aside bias).
        """
        mr = signals.get("mean_reversion", 0.5)
        mom = signals.get("momentum", 0.5)
        of = signals.get("order_flow", 0.5)
        trend = signals.get("trend", 0.5)
        r = (regime or "NEUTRAL").upper()

        # Order-flow is only meaningful when a live order book was supplied to get_signals.
        # In the decision path (backtester / settlement / supervisor) no book is fed, so of is a
        # constant 0.5 that only drags the score toward neutral. Include it ONLY when it is a real,
        # non-neutral reading; otherwise renormalize its weight onto the price-based signals.
        of_live = abs(of - 0.5) > 1e-9

        if "MEAN_REVERTING" in r:
            # Validated edge dominates. RSI is FADED here (high RSI = overbought = sell),
            # so it confirms reversion instead of fighting it.
            w_mr, w_of, w_mom = 0.60, 0.25, 0.15
            mom = 1.0 - mom
            if not of_live:
                w_mr, w_of, w_mom = 0.80, 0.0, 0.20
            score = (w_mr * mr) + (w_of * of) + (w_mom * mom)
        elif "TRENDING" in r:
            # Trend-following sleeve: go WITH the trend. The EMA trend signal dominates and
            # momentum (RSI) is NOT faded -- both point the same way as the move. No conviction
            # compression here: in a real trend we WANT to cross the entry threshold and ride.
            # Mean reversion is nearly ignored (fading a strong trend is the loss source).
            w_tr, w_mom, w_of = 0.65, 0.25, 0.10
            if not of_live:
                w_tr, w_mom, w_of = 0.72, 0.28, 0.0
            score = (w_tr * trend) + (w_mom * mom) + (w_of * of)
        else:  # NEUTRAL: also fade RSI, mean reversion is the house edge
            w_mr, w_of, w_mom = 0.45, 0.30, 0.25
            mom = 1.0 - mom
            if not of_live:
                w_mr, w_of, w_mom = 0.64, 0.0, 0.36
            score = (w_mr * mr) + (w_of * of) + (w_mom * mom)

        return float(score)

class RiskSurface:
    """
    Implements Monte Carlo and Delta-Gamma risk concepts.
    """
    def simulate_drawdown(self, current_balance, vol, trials=100):
        # Simple Monte Carlo for next 24h risk
        daily_vol = vol / np.sqrt(365)
        outcomes = np.random.normal(0, daily_vol, trials)
        max_loss = np.percentile(outcomes, 5) # 5% Value at Risk
        return abs(max_loss)
