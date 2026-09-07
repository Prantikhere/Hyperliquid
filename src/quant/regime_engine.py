import numpy as np
import pandas as pd
from src.utils.logger import log

class RegimeEngine:
    """
    Detects market regimes (Trending vs Mean-Reverting) using statistical metrics.
    In a full production system, this would use HMM (Hidden Markov Models) or GMM.
    For this integration, we implement a robust 'Efficiency Ratio' and 'Volatility' based detector.
    """
    def __init__(self, window=20):
        self.window = window

    def detect_regime(self, prices):
        """
        Input: List or Series of historical prices.
        Output: Regime (TRENDING, MEAN_REVERTING, HIGH_VOLATILITY, or NEUTRAL)
        """
        if len(prices) < self.window:
            return "NEUTRAL"

        try:
            df = pd.Series(prices, dtype=float)
            returns = df.pct_change().dropna()
            if len(returns) < 5:
                return "NEUTRAL"

            # 1. Kaufman Efficiency Ratio (ER) = net directional move / total path length.
            # High ER => clean trend; low ER => choppy / mean-reverting.
            total_change = abs(df.iloc[-1] - df.iloc[0])
            sum_absolute_changes = df.diff().abs().sum()
            er = total_change / sum_absolute_changes if sum_absolute_changes != 0 else 0.0

            # 2. Volatility as a RELATIVE expansion measure, not an absurd annualization.
            # Compare recent realized vol to the full-window baseline. A ratio > 1.5 means
            # volatility is genuinely expanding right now (a real HIGH_VOL condition),
            # instead of the old sqrt(365*4800) scaling that flagged HIGH_VOL every time.
            recent_n = max(5, self.window // 4)
            full_vol = returns.std()
            recent_vol = returns.tail(recent_n).std()
            vol_ratio = (recent_vol / full_vol) if full_vol > 0 else 1.0

            # 3. Base regime from efficiency ratio (thresholds tuned for noisy crypto ticks).
            if er > 0.30:
                regime = "TRENDING"
            elif er < 0.15:
                regime = "MEAN_REVERTING"
            else:
                regime = "NEUTRAL"

            # High-vol flag only when volatility is expanding relative to its own baseline.
            if vol_ratio > 1.5:
                regime += "_HIGH_VOL"

            log.debug(f"Regime Analysis: ER={er:.2f}, VolRatio={vol_ratio:.2f} -> {regime}")
            return regime
            
        except Exception as e:
            log.error(f"Regime Detection Error: {e}")
            return "ERROR"

    def get_dynamic_thresholds(self, prices):
        """ATR-based protective TP/SL with a positive payoff ratio.

        Sizing is anchored to the ATR (average per-bar absolute return) so the stop sits
        outside normal noise. The payoff ratio is >= 2:1 in every regime so expectancy stays
        positive even at a ~50% hit rate -- the previous 1.2:1 mean-reverting ratio was the
        root cause of the negative backtest (many small wins could not pay for the losers).

        NOTE: for mean-reverting/neutral entries the primary exit is the reversion-to-mean
        signal exit in the caller (backtester / settlement); these TP/SL are the backstops.
        """
        if len(prices) < self.window:
            return 0.04, -0.02  # Fallback TP 4.0%, SL 2.0% (2:1)

        try:
            df = pd.Series(prices, dtype=float)
            returns = df.pct_change().dropna()
            # ATR proxy: typical per-bar move over the window (robust vs a single-sigma blowup).
            atr = float(returns.abs().tail(self.window).mean())

            regime = self.detect_regime(prices)
            if "HIGH_VOL" in regime:
                sl = max(0.020, min(0.06, atr * 2.5))  # wider stop in expanding vol
            else:
                sl = max(0.015, min(0.05, atr * 2.0))

            # Payoff ratio by regime. Trends can run further, so give them more room.
            if "TRENDING" in regime:
                tp = sl * 3.0
            else:
                tp = sl * 2.0

            return float(tp), -float(sl)
        except Exception as e:
            log.error(f"Dynamic Threshold Error: {e}")
            return 0.04, -0.02

    def get_strategy_suggestion(self, regime):
        """Maps market regime to recommended quant logic."""
        mapping = {
            "TRENDING": "Trend-following (Moving Average Crossover / Breakout)",
            "MEAN_REVERTING": "Mean-reversion (Bollinger Bands / RSI Overbought-Oversold)",
            "TRENDING_HIGH_VOL": "Volatility Breakout (Donchian Channels)",
            "MEAN_REVERTING_HIGH_VOL": "Scalping / Grid Trading",
            "NEUTRAL": "Market Neutral / Wait for signal"
        }
        return mapping.get(regime, "Neutral")
