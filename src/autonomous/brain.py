"""
Autonomous Brain: Orchestrates all autonomous capabilities.

The Brain is the central nervous system that:
1. Listens to trade events (entry, exit, regime change)
2. Dispatches to appropriate modules (forensics, lifecycle, tuner, regime memory)
3. Coordinates cross-module learning (e.g., forensics feeds lifecycle)
4. Provides unified interface for the trading system
5. Self-monitors and reports its own health

Architecture:
    Trade Event → Brain → [Forensics, Lifecycle, Tuner, RegimeMemory]
                              ↓
                        Actionable Insights
                              ↓
                        System Adaptation
"""
import json
import os
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

from src.autonomous.forensics import PostTradeForensics, TradeForensics
from src.autonomous.lifecycle import StrategyLifecycle
from src.autonomous.tuner import ParameterTuner
from src.autonomous.regime_memory import RegimeMemory

log = logging.getLogger(__name__)


@dataclass
class BrainState:
    """Current state of the autonomous brain."""
    initialized: bool = False
    last_cycle_time: float = 0.0
    total_analyses: int = 0
    total_tunes: int = 0
    total_lessons: int = 0
    # Health
    health_score: float = 1.0
    last_error: str = ""
    error_count: int = 0


class AutonomousBrain:
    """
    The autonomous brain orchestrates all learning and adaptation.
    
    Usage:
        brain = AutonomousBrain(db=db, redis=redis, learning_module=lm)
        
        # On trade entry
        brain.on_trade_entry(symbol, side, regime, confidence, strategy)
        
        # On trade exit
        brain.on_trade_exit(trade_data)
        
        # Each cycle
        brain.cycle()
        
        # Get recommendations
        rec = brain.get_trade_recommendation(symbol, regime)
    """

    def __init__(self, db=None, redis_client=None, learning_module=None):
        self.db = db
        self.redis = redis_client
        self.state = BrainState()
        
        # Initialize modules
        self.forensics = PostTradeForensics(db, redis_client, learning_module)
        self.lifecycle = StrategyLifecycle(db, redis_client)
        self.tuner = ParameterTuner(db, redis_client)
        self.regime_memory = RegimeMemory(db, redis_client)
        
        # Track active trades for context
        self.active_trades: Dict[str, Dict] = {}
        
        # Load previous state
        self._load_state()
        
        self.state.initialized = True
        log.info("[BRAIN] Autonomous Brain initialized")

    def _load_state(self):
        """Load brain state from disk."""
        state_file = "data/brain_state.json"
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
                self.state.total_analyses = data.get("total_analyses", 0)
                self.state.total_tunes = data.get("total_tunes", 0)
                self.state.total_lessons = data.get("total_lessons", 0)
                log.info(f"[BRAIN] Loaded state: {self.state.total_analyses} analyses, "
                         f"{self.state.total_tunes} tunes")
            except Exception as e:
                log.error(f"[BRAIN] Failed to load state: {e}")

    def _save_state(self):
        """Persist brain state to disk."""
        os.makedirs("data", exist_ok=True)
        state_file = "data/brain_state.json"
        data = {
            "total_analyses": self.state.total_analyses,
            "total_tunes": self.state.total_tunes,
            "total_lessons": self.state.total_lessons,
            "last_cycle_time": self.state.last_cycle_time,
            "health_score": self.state.health_score,
            "last_updated": datetime.utcnow().isoformat(),
        }
        with open(state_file, 'w') as f:
            json.dump(data, f, indent=2)

    # ===== EVENT HANDLERS =====

    def on_trade_entry(self, symbol: str, side: str, regime: str,
                       confidence: float, strategy: str = "composite"):
        """Called when a new trade is opened."""
        trade_key = f"{symbol}:{side}:{int(time.time())}"
        self.active_trades[trade_key] = {
            "symbol": symbol,
            "side": side,
            "regime_at_entry": regime,
            "meta_confidence": confidence,
            "strategy": strategy,
            "entry_time": time.time(),
            "entry_price": 0,  # Will be filled from execution
        }
        log.debug(f"[BRAIN] Tracking new trade: {symbol} {side} in {regime}")

    def on_trade_exit(self, trade_data: Dict) -> Optional[TradeForensics]:
        """
        Called when a trade is closed. This is the main learning event.
        
        trade_data should contain:
        - symbol, side, entry_price, exit_price
        - roi_pct, held_seconds
        - regime_at_entry, regime_at_exit
        - meta_confidence, quant_action
        """
        # 1. Run forensics
        forensics = self.forensics.analyze_trade(trade_data)
        self.state.total_analyses += 1
        self.state.total_lessons += 1

        # 2. Record in lifecycle
        strategy = trade_data.get("strategy", "composite")
        regime = trade_data.get("regime_at_entry", "UNKNOWN")
        is_win = trade_data.get("roi_pct", 0) > 0
        self.lifecycle.record_trade(
            symbol=trade_data.get("symbol", ""),
            strategy=strategy,
            regime=regime,
            roi=trade_data.get("roi_pct", 0),
            is_win=is_win
        )

        # 3. Record in regime memory
        self.regime_memory.record_trade(
            symbol=trade_data.get("symbol", ""),
            regime=regime,
            strategy=strategy,
            roi=trade_data.get("roi_pct", 0),
            is_win=is_win
        )

        # 4. Check if symbol should be banned
        if forensics.should_ban_symbol:
            self._ban_symbol(trade_data.get("symbol", ""), forensics.lesson)

        # 5. Cross-module learning
        self._cross_module_learning(forensics)

        # Save state
        self._save_state()

        return forensics

    def _ban_symbol(self, symbol: str, reason: str):
        """Ban a symbol from trading."""
        if self.redis:
            self.redis.sadd("banned_symbols", symbol)
            self.redis.set(f"ban_reason:{symbol}", reason, ex=86400 * 7)
        log.warning(f"[BRAIN] Banned {symbol}: {reason}")

    def _cross_module_learning(self, forensics: TradeForensics):
        """Coordinate learning across modules."""
        # Forensics → Lifecycle: Adjust strategy weights based on failure mode
        if forensics.failure_mode == "REGIME_MISMATCH":
            # Reduce weight for this strategy in this regime
            key = f"{forensics.symbol}:composite:{forensics.regime_at_entry}"
            if key in self.lifecycle.strategies:
                perf = self.lifecycle.strategies[key]
                perf.health_score *= 0.9  # Penalize
                log.debug(f"[BRAIN] Reduced health for {key} due to regime mismatch")

        # Forensics → RegimeMemory: Update regime insights
        if forensics.severity in ("CATASTROPHIC", "MAJOR"):
            self.regime_memory.record_trade(
                symbol=forensics.symbol,
                regime=forensics.regime_at_entry,
                strategy="composite",
                roi=forensics.roi_pct,
                is_win=False
            )

    # ===== CYCLE =====

    def cycle(self):
        """Main brain cycle. Called periodically by hl_executor."""
        now = time.time()
        
        # Don't cycle too often
        if now - self.state.last_cycle_time < 300:  # 5 min minimum
            return

        log.info("[BRAIN] Starting cycle...")

        try:
            # 1. Check if tuning is needed
            if self.tuner.should_tune():
                adjustments = self.tuner.tune()
                if adjustments:
                    self.state.total_tunes += 1
                    log.info(f"[BRAIN] Tuned parameters: {adjustments}")

            # 2. Apply daily decay to lifecycle
            self.lifecycle.apply_daily_decay()

            # 3. Check for retirement candidates
            retired = self.lifecycle.get_retirement_candidates()
            if retired:
                for symbol in retired:
                    self._ban_symbol(symbol, "Strategy lifecycle retirement")
                log.info(f"[BRAIN] Retired {len(retired)} symbols: {retired}")

            # 4. Update brain health
            self._update_health()

            self.state.last_cycle_time = now
            self._save_state()

            log.info(f"[BRAIN] Cycle complete. Health={self.state.health_score:.2f} "
                     f"Analyses={self.state.total_analyses} Tunes={self.state.total_tunes}")

        except Exception as e:
            self.state.last_error = str(e)
            self.state.error_count += 1
            log.error(f"[BRAIN] Cycle error: {e}")

    def _update_health(self):
        """Calculate brain health score."""
        # Health factors
        error_penalty = min(0.3, self.state.error_count * 0.05)
        recency_bonus = 0.2 if (time.time() - self.state.last_cycle_time) < 3600 else 0
        
        self.state.health_score = max(0, min(1, 
            1.0 - error_penalty + recency_bonus
        ))

    # ===== RECOMMENDATIONS =====

    def get_trade_recommendation(self, symbol: str, regime: str) -> Dict:
        """
        Get autonomous recommendation for trading a symbol.
        
        Returns:
        - should_trade: bool
        - confidence_adjustment: float (multiplier)
        - reason: str
        - strategy_weight: float
        """
        should_trade = True
        confidence_adj = 1.0
        reason = "No autonomous signal"
        strategy_weight = 1.0

        # Check forensics
        health = self.forensics.get_symbol_health(symbol)
        if health.get("status") == "banned":
            should_trade = False
            reason = f"Banned: {health.get('last_failure_mode', 'unknown')}"
        elif health.get("confidence_adjustment", 1.0) != 1.0:
            confidence_adj *= health["confidence_adjustment"]
            reason = f"Adjusted: {health.get('last_failure_mode', 'unknown')}"

        # Check lifecycle
        lifecycle_health = self.lifecycle.get_symbol_health(symbol)
        if lifecycle_health < 0.3:
            should_trade = False
            reason = f"Lifecycle health low: {lifecycle_health:.2f}"
        elif lifecycle_health < 0.5:
            confidence_adj *= 0.8
            reason = f"Lifecycle health moderate: {lifecycle_health:.2f}"

        # Check regime memory
        regime_ok, regime_reason = self.regime_memory.should_trade_symbol(symbol, regime)
        if not regime_ok:
            should_trade = False
            reason = regime_reason

        # Check tuner parameters
        tuner_params = self.tuner.get_status()
        regime_adj = self.tuner.get_regime_adjustment(regime)

        return {
            "should_trade": should_trade,
            "confidence_adjustment": confidence_adj,
            "reason": reason,
            "strategy_weight": strategy_weight,
            "symbol_health": health,
            "lifecycle_health": lifecycle_health,
            "regime_recommendations": self.regime_memory.get_recommendations(symbol, regime),
            "tuner_params": tuner_params,
            "regime_adjustments": regime_adj,
        }

    # ===== REPORTING =====

    def get_report(self) -> Dict:
        """Get comprehensive brain status report."""
        return {
            "state": {
                "health": self.state.health_score,
                "total_analyses": self.state.total_analyses,
                "total_tunes": self.state.total_tunes,
                "total_lessons": self.state.total_lessons,
                "last_cycle": datetime.fromtimestamp(self.state.last_cycle_time).isoformat() if self.state.last_cycle_time > 0 else "never",
                "errors": self.state.error_count,
            },
            "forensics": {
                "failure_modes": self.forensics.get_failure_mode_stats(),
                "recent_lessons": self.forensics.get_all_lessons(5),
            },
            "lifecycle": self.lifecycle.get_strategy_report(),
            "tuner": self.tuner.get_status(),
            "regime_memory": self.regime_memory.get_memory_report(),
        }

    def log_report(self):
        """Log a human-readable report."""
        report = self.get_report()
        
        log.info("=== AUTONOMOUS BRAIN REPORT ===")
        log.info(f"Health: {report['state']['health']:.2f}")
        log.info(f"Analyses: {report['state']['total_analyses']}")
        log.info(f"Tunes: {report['state']['total_tunes']}")
        log.info(f"Lessons: {report['state']['total_lessons']}")
        
        fm = report['forensics']['failure_modes']
        if fm:
            log.info(f"Failure modes: {json.dumps(fm)}")
        
        lifecycle = report['lifecycle']
        log.info(f"Strategies tracked: {lifecycle.get('total_strategies', 0)}")
        log.info(f"Overall win rate: {lifecycle.get('overall_win_rate', 0):.0%}")
        
        if lifecycle.get('best'):
            log.info(f"Best: {lifecycle['best'][0]}")
        if lifecycle.get('worst'):
            log.info(f"Worst: {lifecycle['worst'][0]}")
        
        tuner = report['tuner']
        log.info(f"Parameters: buy={tuner['buy_threshold']:.4f} sell={tuner['sell_threshold']:.4f} "
                 f"conf={tuner['min_confidence']:.4f}")
        
        log.info("=== END REPORT ===")
