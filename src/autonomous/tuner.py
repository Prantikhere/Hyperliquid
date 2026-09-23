"""
Parameter Tuner: Auto-adjusts trading parameters based on performance.

Responsibilities:
1. Monitor win rate, Sharpe, drawdown over rolling windows
2. Auto-adjust buy_threshold, sell_threshold, min_confidence
3. Detect regime shifts and adapt parameters
4. Prevent over-tuning by limiting adjustment frequency
5. Write updated parameters to strategy_config.json
"""
import json
import os
import time
import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class TuningState:
    """Current state of the parameter tuner."""
    last_tune_time: float = 0.0
    tune_count: int = 0
    current_buy_threshold: float = 0.55
    current_sell_threshold: float = 0.45
    current_min_confidence: float = 0.50
    # Performance tracking
    rolling_win_rate: float = 0.5
    rolling_sharpe: float = 0.0
    rolling_avg_roi: float = 0.0
    drawdown_pct: float = 0.0
    # Regime-specific adjustments
    regime_adjustments: Dict = None

    def __post_init__(self):
        if self.regime_adjustments is None:
            self.regime_adjustments = {}


class ParameterTuner:
    """Auto-tunes trading parameters based on rolling performance."""

    # Tuning constraints
    MIN_TUNE_INTERVAL_HOURS = 4  # Don't tune more often than every 4h
    MAX_TUNE_PER_DAY = 6         # Max 6 tuning events per day
    MIN_TRADES_FOR_TUNING = 10   # Need at least 10 trades to tune
    MAX_ADJUSTMENT_PER_STEP = 0.05  # Max 5% change per tuning step

    # Parameter bounds
    BUY_THRESHOLD_RANGE = (0.50, 0.70)
    SELL_THRESHOLD_RANGE = (0.30, 0.50)
    CONFIDENCE_RANGE = (0.40, 0.65)

    def __init__(self, db=None, redis_client=None, config_path="models_local/strategy_config.json"):
        self.db = db
        self.redis = redis_client
        self.config_path = config_path
        self.state_file = "data/tuner_state.json"
        self.state = TuningState()
        self._load_state()

    def _load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                for k, v in data.items():
                    if hasattr(self.state, k):
                        setattr(self.state, k, v)
                # DRIFT GUARD: strategy_config.json is the live source of truth
                # (Supervisor loads it every boot). If tuner state diverged from
                # config (e.g. config restored from git, or tuner wrote only state),
                # re-anchor thresholds to config so the next tune steps from live
                # values instead of a ghost baseline.
                if os.path.exists(self.config_path):
                    try:
                        with open(self.config_path, 'r') as f:
                            cfg = json.load(f)
                        for key, attr in (
                            ("buy_threshold", "current_buy_threshold"),
                            ("sell_threshold", "current_sell_threshold"),
                            ("min_confidence", "current_min_confidence"),
                        ):
                            cfg_val = cfg.get(key)
                            if cfg_val is not None and abs(float(cfg_val) - float(getattr(self.state, attr))) > 1e-9:
                                log.warning(
                                    f"[TUNER] Drift: state.{attr}={getattr(self.state, attr)} "
                                    f"vs config.{key}={cfg_val} -- re-anchoring to config"
                                )
                                setattr(self.state, attr, float(cfg_val))
                    except Exception as cfg_err:
                        log.debug(f"[TUNER] Drift guard skipped (config unreadable): {cfg_err}")
                log.info(f"[TUNER] Loaded state: {self.state.tune_count} previous tunes "
                         f"(buy={self.state.current_buy_threshold} sell={self.state.current_sell_threshold} "
                         f"conf={self.state.current_min_confidence})")
            except Exception as e:
                log.error(f"[TUNER] Failed to load state: {e}")

    def _save_state(self):
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        data = {
            'last_tune_time': self.state.last_tune_time,
            'tune_count': self.state.tune_count,
            'current_buy_threshold': self.state.current_buy_threshold,
            'current_sell_threshold': self.state.current_sell_threshold,
            'current_min_confidence': self.state.current_min_confidence,
            'rolling_win_rate': self.state.rolling_win_rate,
            'rolling_sharpe': self.state.rolling_sharpe,
            'rolling_avg_roi': self.state.rolling_avg_roi,
            'drawdown_pct': self.state.drawdown_pct,
            'regime_adjustments': self.state.regime_adjustments,
        }
        with open(self.state_file, 'w') as f:
            json.dump(data, f, indent=2)

    def should_tune(self) -> bool:
        """Check if it's time to tune parameters."""
        now = time.time()
        hours_since_last = (now - self.state.last_tune_time) / 3600

        # Don't tune too frequently
        if hours_since_last < self.MIN_TUNE_INTERVAL_HOURS:
            return False

        # Check daily limit
        today_start = datetime.now().replace(hour=0, minute=0, second=0).timestamp()
        if self.state.last_tune_time > today_start:
            # Already tuned today, check count
            if self.state.tune_count >= self.MAX_TUNE_PER_DAY:
                return False

        return True

    def analyze_performance(self) -> Dict:
        """Analyze recent trading performance from calibration ledger."""
        lessons_file = "data/trade_lessons.jsonl"
        if not os.path.exists(lessons_file):
            return {"trades": 0, "win_rate": 0.5, "sharpe": 0, "avg_roi": 0}

        recent_trades = []
        cutoff = time.time() - (7 * 86400)  # Last 7 days

        with open(lessons_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    trade = json.loads(line)
                    ts = datetime.fromisoformat(trade.get('timestamp', '')).timestamp()
                    if ts > cutoff:
                        recent_trades.append(trade)
                except (json.JSONDecodeError, ValueError):
                    continue

        if len(recent_trades) < self.MIN_TRADES_FOR_TUNING:
            return {"trades": len(recent_trades), "win_rate": 0.5, "sharpe": 0, "avg_roi": 0}

        # Calculate metrics
        rois = [t.get('roi_pct', 0) for t in recent_trades]
        wins = sum(1 for r in rois if r > 0)
        win_rate = wins / len(rois)
        avg_roi = sum(rois) / len(rois)

        # Sharpe ratio (simplified)
        if len(rois) > 1:
            mean_roi = sum(rois) / len(rois)
            variance = sum((r - mean_roi) ** 2 for r in rois) / (len(rois) - 1)
            std_dev = math.sqrt(variance) if variance > 0 else 1
            sharpe = (mean_roi / std_dev) if std_dev > 0 else 0
        else:
            sharpe = 0

        # Drawdown
        cumulative = 0
        peak = 0
        max_dd = 0
        for r in rois:
            cumulative += r
            if cumulative > peak:
                peak = cumulative
            dd = peak - cumulative
            if dd > max_dd:
                max_dd = dd

        return {
            "trades": len(recent_trades),
            "win_rate": win_rate,
            "sharpe": sharpe,
            "avg_roi": avg_roi,
            "drawdown": max_dd,
        }

    def tune(self, force: bool = False) -> Optional[Dict]:
        """
        Perform parameter tuning. Returns the adjustments made, or None if no tuning needed.
        """
        if not force and not self.should_tune():
            return None

        perf = self.analyze_performance()

        if perf["trades"] < self.MIN_TRADES_FOR_TUNING:
            log.info(f"[TUNER] Only {perf['trades']} trades in window, need {self.MIN_TRADES_FOR_TUNING}")
            return None

        # Store current performance metrics
        self.state.rolling_win_rate = perf["win_rate"]
        self.state.rolling_sharpe = perf["sharpe"]
        self.state.rolling_avg_roi = perf["avg_roi"]
        self.state.drawdown_pct = perf.get("drawdown", 0)

        adjustments = {}

        # === BUY THRESHOLD TUNING ===
        # If win rate is low, raise buy threshold (be more selective)
        # If win rate is high, lower buy threshold (more opportunities)
        if perf["win_rate"] < 0.40:
            delta = min(self.MAX_ADJUSTMENT_PER_STEP, (0.40 - perf["win_rate"]) * 0.3)
            new_val = self.state.current_buy_threshold + delta
            new_val = min(new_val, self.BUY_THRESHOLD_RANGE[1])
            if new_val != self.state.current_buy_threshold:
                adjustments["buy_threshold"] = {"old": self.state.current_buy_threshold, "new": new_val}
                self.state.current_buy_threshold = new_val
        elif perf["win_rate"] > 0.65:
            delta = min(self.MAX_ADJUSTMENT_PER_STEP, (perf["win_rate"] - 0.65) * 0.3)
            new_val = self.state.current_buy_threshold - delta
            new_val = max(new_val, self.BUY_THRESHOLD_RANGE[0])
            if new_val != self.state.current_buy_threshold:
                adjustments["buy_threshold"] = {"old": self.state.current_buy_threshold, "new": new_val}
                self.state.current_buy_threshold = new_val

        # === SELL THRESHOLD TUNING (mirror of buy) ===
        if perf["win_rate"] < 0.40:
            delta = min(self.MAX_ADJUSTMENT_PER_STEP, (0.40 - perf["win_rate"]) * 0.3)
            new_val = self.state.current_sell_threshold - delta
            new_val = max(new_val, self.SELL_THRESHOLD_RANGE[0])
            if new_val != self.state.current_sell_threshold:
                adjustments["sell_threshold"] = {"old": self.state.current_sell_threshold, "new": new_val}
                self.state.current_sell_threshold = new_val
        elif perf["win_rate"] > 0.65:
            delta = min(self.MAX_ADJUSTMENT_PER_STEP, (perf["win_rate"] - 0.65) * 0.3)
            new_val = self.state.current_sell_threshold + delta
            new_val = min(new_val, self.SELL_THRESHOLD_RANGE[1])
            if new_val != self.state.current_sell_threshold:
                adjustments["sell_threshold"] = {"old": self.state.current_sell_threshold, "new": new_val}
                self.state.current_sell_threshold = new_val

        # === CONFIDENCE THRESHOLD TUNING ===
        # During drawdowns, raise confidence (be more selective)
        if self.state.drawdown_pct > 3.0:
            delta = min(self.MAX_ADJUSTMENT_PER_STEP, self.state.drawdown_pct * 0.01)
            new_val = self.state.current_min_confidence + delta
            new_val = min(new_val, self.CONFIDENCE_RANGE[1])
            if new_val != self.state.current_min_confidence:
                adjustments["min_confidence"] = {"old": self.state.current_min_confidence, "new": new_val}
                self.state.current_min_confidence = new_val
        elif self.state.drawdown_pct < 1.0 and perf["win_rate"] > 0.60:
            # Good performance, can lower confidence slightly
            delta = min(self.MAX_ADJUSTMENT_PER_STEP, 0.02)
            new_val = self.state.current_min_confidence - delta
            new_val = max(new_val, self.CONFIDENCE_RANGE[0])
            if new_val != self.state.current_min_confidence:
                adjustments["min_confidence"] = {"old": self.state.current_min_confidence, "new": new_val}
                self.state.current_min_confidence = new_val

        if adjustments:
            # Write to config
            self._write_config()
            
            # Update state
            self.state.last_tune_time = time.time()
            self.state.tune_count += 1
            self._save_state()

            # Log
            log.info(f"[TUNER] Tuned {len(adjustments)} parameters: {json.dumps(adjustments)}")
            log.info(f"[TUNER] Performance: WR={perf['win_rate']:.0%} Sharpe={perf['sharpe']:.2f} "
                     f"ROI={perf['avg_roi']:.2f}% DD={self.state.drawdown_pct:.1f}%")

            # Update Redis
            if self.redis:
                self.redis.hset("tuner:last_tune", mapping={
                    "timestamp": str(int(time.time())),
                    "adjustments": json.dumps(adjustments),
                    "win_rate": f"{perf['win_rate']:.4f}",
                    "sharpe": f"{perf['sharpe']:.4f}",
                })

            return adjustments

        log.info(f"[TUNER] No adjustments needed. WR={perf['win_rate']:.0%} Sharpe={perf['sharpe']:.2f}")
        return None

    def _write_config(self):
        """Write current parameters to strategy_config.json."""
        try:
            config = {}
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)

            config["buy_threshold"] = round(self.state.current_buy_threshold, 4)
            config["sell_threshold"] = round(self.state.current_sell_threshold, 4)
            config["min_confidence"] = round(self.state.current_min_confidence, 4)

            # Add tuning metadata
            config["_tuned_at"] = datetime.utcnow().isoformat()
            config["_tune_count"] = self.state.tune_count
            config["_rolling_win_rate"] = round(self.state.rolling_win_rate, 4)
            config["_rolling_sharpe"] = round(self.state.rolling_sharpe, 4)

            with open(self.config_path, 'w') as f:
                json.dump(config, f, indent=2)

            log.info(f"[TUNER] Wrote config: buy={self.state.current_buy_threshold:.4f} "
                     f"sell={self.state.current_sell_threshold:.4f} "
                     f"conf={self.state.current_min_confidence:.4f}")

        except Exception as e:
            log.error(f"[TUNER] Failed to write config: {e}")

    def get_regime_adjustment(self, regime: str) -> Dict:
        """Get regime-specific parameter adjustments."""
        defaults = {
            "TRENDING": {"buy_threshold": 0.02, "sell_threshold": 0.02, "min_confidence": -0.03},
            "MEAN_REVERTING": {"buy_threshold": -0.02, "sell_threshold": -0.02, "min_confidence": 0.02},
            "NEUTRAL": {"buy_threshold": 0.0, "sell_threshold": 0.0, "min_confidence": 0.05},
            "TRENDING_HIGH_VOL": {"buy_threshold": 0.03, "sell_threshold": 0.03, "min_confidence": 0.05},
            "MEAN_REVERTING_HIGH_VOL": {"buy_threshold": -0.01, "sell_threshold": -0.01, "min_confidence": 0.03},
        }
        return defaults.get(regime, defaults["NEUTRAL"])

    def get_status(self) -> Dict:
        """Get current tuner status."""
        return {
            "buy_threshold": self.state.current_buy_threshold,
            "sell_threshold": self.state.current_sell_threshold,
            "min_confidence": self.state.current_min_confidence,
            "tune_count": self.state.tune_count,
            "last_tune": datetime.fromtimestamp(self.state.last_tune_time).isoformat() if self.state.last_tune_time > 0 else "never",
            "rolling_win_rate": self.state.rolling_win_rate,
            "rolling_sharpe": self.state.rolling_sharpe,
            "drawdown_pct": self.state.drawdown_pct,
        }
