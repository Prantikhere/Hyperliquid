import numpy as np
import pandas as pd
from src.utils.logger import log
from src.quant.rnn_predictor import rnn_predictor

class StrategyEnsemble:
    """
    Implements multiple quantitative strategies discussed in the blueprints.
    Provides signals for Mean Reversion, Momentum, and Order Flow.
    """
    def __init__(self):
        self.last_rnn_result = None  # Cache for meta-learner to avoid double RNN call

    def _kalman_filter(self, prices):
        """1D State-Space Kalman Filter: returns (filtered_prices, velocities)."""
        if len(prices) < 5:
            return np.array(prices), np.zeros(len(prices))
        q = 1e-4  # process noise variance
        r = 1e-3  # measurement noise variance
        f = np.array([[1.0, 1.0], [0.0, 1.0]])  # state transition
        h = np.array([[1.0, 0.0]])  # measurement matrix
        
        x = np.array([[prices[0]], [0.0]])
        p = np.eye(2)
        filtered = []
        vels = []
        for z in prices:
            x_pred = f @ x
            p_pred = f @ p @ f.T + np.eye(2) * q
            y = z - (h @ x_pred)[0, 0]
            s = (h @ p_pred @ h.T)[0, 0] + r
            k = p_pred @ h.T / s
            x = x_pred + k * y
            p = (np.eye(2) - k @ h) @ p_pred
            filtered.append(x[0, 0])
            vels.append(x[1, 0])
        return np.array(filtered), np.array(vels)

    def _estimate_garch_vol(self, prices):
        """Recursive GARCH(1,1) conditional volatility clustering."""
        if len(prices) < 5:
            return 0.01
        returns = np.diff(prices) / prices[:-1]
        omega, alpha, beta = 1e-5, 0.15, 0.80
        sigma2 = np.var(returns) if len(returns) > 1 else 1e-4
        for ret in returns:
            sigma2 = omega + alpha * (ret ** 2) + beta * sigma2
        return float(np.sqrt(sigma2))

    def get_signals(self, prices, book=None):
        if len(prices) < 20:
            return {"mean_reversion": 0.5, "momentum": 0.5, "order_flow": 0.5, "trend": 0.5, "rnn": 0.5, "kalman_velocity": 0.5, "garch_vol": 0.01}

        df = pd.Series(prices)
        
        # Kalman Filtering on raw price ticks
        filt_prices, vels = self._kalman_filter(prices)
        curr_filt = filt_prices[-1]
        curr_vel = vels[-1]
        garch_vol = self._estimate_garch_vol(prices)
        
        # 1. Kalman-Enhanced Mean Reversion Signal
        # Measures deviation from the true Kalman-estimated state normalized by GARCH volatility
        price_diff = df.iloc[-1] - curr_filt
        vol_normalizer = (curr_filt * garch_vol) if garch_vol > 0 else df.std()
        kalman_z = price_diff / vol_normalizer if vol_normalizer > 0 else 0
        mr_signal = 1.0 / (1.0 + np.exp(kalman_z)) 

        # 2. Kalman Velocity Momentum Signal
        # Velocity > 0 indicates upward momentum; normalized via sigmoid
        vel_norm = curr_vel / (curr_filt * garch_vol + 1e-9)
        kalman_vel_sig = float(1.0 / (1.0 + np.exp(-5.0 * vel_norm)))

        # 3. Standard Momentum Signal (RSI)
        change = df.diff()
        gain = (change.where(change > 0, 0)).rolling(window=14).mean()
        loss = (-change.where(change < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs)).iloc[-1]
        mom_signal = rsi / 100.0 # High RSI = High Momentum (0 to 1)

        # 4. Order Flow Signal (Bid/Ask Imbalance)
        of_signal = 0.5
        if book:
            bids = sum([float(b[1]) for b in book.get('bids', [])[:5]])
            asks = sum([float(a[1]) for a in book.get('asks', [])[:5]])
            if (bids + asks) > 0:
                of_signal = bids / (bids + asks) # High = Buy Pressure

        # 5. Trend Signal (Kalman Price EMA Separation)
        trend_signal = 0.5
        if len(filt_prices) >= 30:
            filt_series = pd.Series(filt_prices)
            ema_fast = filt_series.ewm(span=10, adjust=False).mean().iloc[-1]
            ema_slow = filt_series.ewm(span=30, adjust=False).mean().iloc[-1]
            if ema_slow != 0:
                rel = (ema_fast - ema_slow) / ema_slow
                trend_signal = float(1.0 / (1.0 + np.exp(-50.0 * rel)))

        # 6. RNN Signal (Neural network price prediction)
        rnn_signal = 0.5
        self.last_rnn_result = None  # Reset cache
        if len(prices) > 30:
            rnn_result = rnn_predictor.predict(prices)
            rnn_signal = rnn_result['prediction']
            self.last_rnn_result = rnn_result  # Cache for meta-learner
            log.debug(f"[RNN] Signal: {rnn_result['signal']}, Confidence: {rnn_result['confidence']:.2f}")

        return {
            "mean_reversion": float(mr_signal),
            "momentum": float(mom_signal),
            "order_flow": float(of_signal),
            "trend": float(trend_signal),
            "rnn": float(rnn_signal),
            "kalman_velocity": float(kalman_vel_sig),
            "garch_vol": float(garch_vol)
        }

    def composite_score(self, signals, regime="NEUTRAL"):
        """
        Regime-aware directional score in [0, 1] (>0.5 bullish, <0.5 bearish).
        Enhanced with Kalman filter velocity and GARCH volatility normalization.
        """
        mr = signals.get("mean_reversion", 0.5)
        mom = signals.get("momentum", 0.5)
        of = signals.get("order_flow", 0.5)
        trend = signals.get("trend", 0.5)
        rnn = signals.get("rnn", 0.5)
        k_vel = signals.get("kalman_velocity", 0.5)
        r = (regime or "NEUTRAL").upper()

        of_live = abs(of - 0.5) > 1e-9

        if "MEAN_REVERTING" in r:
            # Reversion dominant: Kalman-normalized reversion + Kalman velocity inflection confirmation
            w_mr, w_kvel, w_mom, w_rnn = 0.50, 0.25, 0.15, 0.10
            mom = 1.0 - mom  # fade RSI
            score = (w_mr * mr) + (w_kvel * k_vel) + (w_mom * mom) + (w_rnn * rnn)
        elif "TRENDING" in r:
            # Trend-following sleeve: Kalman EMA trend + Kalman velocity confirmation
            w_tr, w_kvel, w_mom, w_rnn = 0.45, 0.30, 0.15, 0.10
            score = (w_tr * trend) + (w_kvel * k_vel) + (w_mom * mom) + (w_rnn * rnn)
        elif "HIGH_VOL" in r:
            # High volatility expansion: follow dominant trend and velocity with RNN confirmation
            w_tr, w_kvel, w_mom, w_rnn = 0.40, 0.30, 0.20, 0.10
            score = (w_tr * trend) + (w_kvel * k_vel) + (w_mom * mom) + (w_rnn * rnn)
        else:  # NEUTRAL: fade RSI, Kalman mean reversion + velocity inflection
            w_mr, w_kvel, w_mom, w_rnn = 0.45, 0.25, 0.20, 0.10
            mom = 1.0 - mom
            score = (w_mr * mr) + (w_kvel * k_vel) + (w_mom * mom) + (w_rnn * rnn)

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
