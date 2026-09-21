"""
Strategy Lifecycle Manager: Tracks per-strategy/symbol performance over time.

Responsibilities:
1. Track P&L per strategy per symbol per regime
2. Auto-retire strategies that consistently lose
3. Auto-promote strategies that consistently win
4. Adjust strategy weights based on recent performance
5. Provide health scores for dynamic whitelist
"""
import json
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class StrategyPerformance:
    """Performance metrics for a strategy on a symbol in a regime."""
    symbol: str
    strategy: str  # mean_reversion, momentum, trend, rnn, composite
    regime: str
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_roi: float = 0.0
    avg_roi: float = 0.0
    win_rate: float = 0.0
    sharpe: float = 0.0
    max_drawdown: float = 0.0
    last_trade_time: float = 0.0
    health_score: float = 0.5  # 0.0 = dead, 1.0 = excellent


class StrategyLifecycle:
    """Manages strategy lifecycle: birth, health, retirement."""

    # Health thresholds
    MIN_TRADES_FOR_EVALUATION = 5
    RETIREMENT_WIN_RATE = 0.30  # Below this = retire
    RETIREMENT_SHARPE = -0.5    # Below this = retire
    PROMOTION_WIN_RATE = 0.60   # Above this = promote
    PROMOTION_SHARPE = 1.0      # Above this = promote
    HEALTH_DECAY_RATE = 0.95    # Daily decay towards neutral

    def __init__(self, db=None, redis_client=None):
        self.db = db
        self.redis = redis_client
        self.state_file = "data/strategy_lifecycle.json"
        self.strategies: Dict[str, StrategyPerformance] = {}
        self._load_state()

    def _load_state(self):
        """Load strategy state from disk."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, 'r') as f:
                    data = json.load(f)
                for key, vals in data.items():
                    self.strategies[key] = StrategyPerformance(**vals)
                log.info(f"[LIFECYCLE] Loaded {len(self.strategies)} strategy records")
            except Exception as e:
                log.error(f"[LIFECYCLE] Failed to load state: {e}")

    def _save_state(self):
        """Persist strategy state to disk."""
        os.makedirs(os.path.dirname(self.state_file), exist_ok=True)
        data = {}
        for key, perf in self.strategies.items():
            data[key] = {
                'symbol': perf.symbol,
                'strategy': perf.strategy,
                'regime': perf.regime,
                'total_trades': perf.total_trades,
                'wins': perf.wins,
                'losses': perf.losses,
                'total_roi': perf.total_roi,
                'avg_roi': perf.avg_roi,
                'win_rate': perf.win_rate,
                'sharpe': perf.sharpe,
                'max_drawdown': perf.max_drawdown,
                'last_trade_time': perf.last_trade_time,
                'health_score': perf.health_score,
            }
        with open(self.state_file, 'w') as f:
            json.dump(data, f, indent=2)

    def record_trade(self, symbol: str, strategy: str, regime: str,
                     roi: float, is_win: bool):
        """Record a trade outcome for a strategy."""
        key = f"{symbol}:{strategy}:{regime}"
        
        if key not in self.strategies:
            self.strategies[key] = StrategyPerformance(
                symbol=symbol, strategy=strategy, regime=regime
            )

        perf = self.strategies[key]
        perf.total_trades += 1
        if is_win:
            perf.wins += 1
        else:
            perf.losses += 1
        perf.total_roi += roi
        perf.avg_roi = perf.total_roi / perf.total_trades
        perf.win_rate = perf.wins / perf.total_trades if perf.total_trades > 0 else 0
        perf.last_trade_time = time.time()

        # Update health score
        perf.health_score = self._calculate_health(perf)

        # Persist
        self._save_state()

        # Update Redis
        self._update_redis(key, perf)

        log.info(f"[LIFECYCLE] {symbol} {strategy} in {regime}: "
                 f"trades={perf.total_trades} WR={perf.win_rate:.0%} "
                 f"ROI={perf.avg_roi:.2f}% health={perf.health_score:.2f}")

    def _calculate_health(self, perf: StrategyPerformance) -> float:
        """Calculate health score (0.0 = dead, 1.0 = excellent)."""
        if perf.total_trades < self.MIN_TRADES_FOR_EVALUATION:
            return 0.5  # Neutral until enough data

        # Base score from win rate (0-1)
        wr_score = perf.win_rate

        # Sharpe bonus/penalty
        sharpe_score = max(0, min(1, (perf.sharpe + 1) / 3))  # Map [-1,2] to [0,1]

        # ROI trend (are recent trades getting better?)
        roi_score = max(0, min(1, (perf.avg_roi + 5) / 10))  # Map [-5,5] to [0,1]

        # Time decay (recent trades matter more)
        hours_since_last = (time.time() - perf.last_trade_time) / 3600
        time_factor = max(0.5, 1.0 - (hours_since_last / 168))  # Decay over 1 week

        # Weighted combination
        health = (
            0.40 * wr_score +
            0.25 * sharpe_score +
            0.25 * roi_score +
            0.10 * time_factor
        )

        return max(0.0, min(1.0, health))

    def _update_redis(self, key: str, perf: StrategyPerformance):
        """Update Redis with strategy performance."""
        if not self.redis:
            return

        redis_key = f"strategy_perf:{key}"
        self.redis.hset(redis_key, mapping={
            "symbol": perf.symbol,
            "strategy": perf.strategy,
            "regime": perf.regime,
            "total_trades": str(perf.total_trades),
            "wins": str(perf.wins),
            "losses": str(perf.losses),
            "win_rate": f"{perf.win_rate:.4f}",
            "avg_roi": f"{perf.avg_roi:.4f}",
            "health": f"{perf.health_score:.4f}",
        })
        self.redis.expire(redis_key, 86400 * 30)

    def should_retire(self, symbol: str, strategy: str, regime: str) -> bool:
        """Check if a strategy should be retired for this symbol/regime."""
        key = f"{symbol}:{strategy}:{regime}"
        perf = self.strategies.get(key)

        if not perf or perf.total_trades < self.MIN_TRADES_FOR_EVALUATION:
            return False

        return (perf.win_rate < self.RETIREMENT_WIN_RATE and 
                perf.sharpe < self.RETIREMENT_SHARPE)

    def should_promote(self, symbol: str, strategy: str, regime: str) -> bool:
        """Check if a strategy should be promoted for this symbol/regime."""
        key = f"{symbol}:{strategy}:{regime}"
        perf = self.strategies.get(key)

        if not perf or perf.total_trades < self.MIN_TRADES_FOR_EVALUATION:
            return False

        return (perf.win_rate > self.PROMOTION_WIN_RATE and 
                perf.sharpe > self.PROMOTION_SHARPE)

    def get_strategy_weight(self, symbol: str, strategy: str, regime: str) -> float:
        """Get weight multiplier for a strategy based on health."""
        key = f"{symbol}:{strategy}:{regime}"
        perf = self.strategies.get(key)

        if not perf or perf.total_trades < self.MIN_TRADES_FOR_EVALUATION:
            return 1.0  # Default weight for unknown strategies

        # Map health to weight: 0.0 health = 0.3x weight, 1.0 health = 1.5x weight
        return 0.3 + (perf.health_score * 1.2)

    def get_symbol_health(self, symbol: str) -> float:
        """Get overall health score for a symbol across all strategies/regimes."""
        relevant = [p for p in self.strategies.values() if p.symbol == symbol]
        if not relevant:
            return 0.5

        # Average health, weighted by trade count
        total_weight = sum(p.total_trades for p in relevant)
        if total_weight == 0:
            return 0.5

        weighted_health = sum(p.health_score * p.total_trades for p in relevant)
        return weighted_health / total_weight

    def get_worst_strategies(self, n: int = 5) -> List[StrategyPerformance]:
        """Get the N worst performing strategy-symbol-regime combinations."""
        valid = [p for p in self.strategies.values() 
                 if p.total_trades >= self.MIN_TRADES_FOR_EVALUATION]
        return sorted(valid, key=lambda p: p.health_score)[:n]

    def get_best_strategies(self, n: int = 5) -> List[StrategyPerformance]:
        """Get the N best performing strategy-symbol-regime combinations."""
        valid = [p for p in self.strategies.values() 
                 if p.total_trades >= self.MIN_TRADES_FOR_EVALUATION]
        return sorted(valid, key=lambda p: p.health_score, reverse=True)[:n]

    def get_retirement_candidates(self) -> List[str]:
        """Get list of symbols that should be banned from trading."""
        candidates = set()
        for key, perf in self.strategies.items():
            if self.should_retire(perf.symbol, perf.strategy, perf.regime):
                candidates.add(perf.symbol)
        return list(candidates)

    def get_strategy_report(self) -> Dict:
        """Get comprehensive strategy performance report."""
        report = {
            "total_strategies": len(self.strategies),
            "total_trades": sum(p.total_trades for p in self.strategies.values()),
            "overall_win_rate": 0,
            "best": [],
            "worst": [],
            "retirement_candidates": self.get_retirement_candidates(),
        }

        total_wins = sum(p.wins for p in self.strategies.values())
        total_trades = report["total_trades"]
        report["overall_win_rate"] = total_wins / total_trades if total_trades > 0 else 0

        best = self.get_best_strategies(5)
        worst = self.get_worst_strategies(5)

        for p in best:
            report["best"].append({
                "symbol": p.symbol, "strategy": p.strategy, "regime": p.regime,
                "win_rate": f"{p.win_rate:.0%}", "avg_roi": f"{p.avg_roi:.2f}%",
                "health": f"{p.health_score:.2f}", "trades": p.total_trades
            })

        for p in worst:
            report["worst"].append({
                "symbol": p.symbol, "strategy": p.strategy, "regime": p.regime,
                "win_rate": f"{p.win_rate:.0%}", "avg_roi": f"{p.avg_roi:.2f}%",
                "health": f"{p.health_score:.2f}", "trades": p.total_trades
            })

        return report

    def apply_daily_decay(self):
        """Apply daily health decay to prevent stale data from being trusted."""
        now = time.time()
        for key, perf in self.strategies.items():
            hours_since = (now - perf.last_trade_time) / 3600
            if hours_since > 24:
                # Decay health towards 0.5 (neutral)
                days_stale = hours_since / 24
                decay = self.HEALTH_DECAY_RATE ** days_stale
                perf.health_score = 0.5 + (perf.health_score - 0.5) * decay

        self._save_state()
        log.info(f"[LIFECYCLE] Applied daily decay to {len(self.strategies)} strategies")
