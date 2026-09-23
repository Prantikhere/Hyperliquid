"""
Regime Memory: Remembers what works in each market regime.

Responsibilities:
1. Track which strategies/symbols perform well in each regime
2. Remember regime transitions and their outcomes
3. Provide regime-specific trading recommendations
4. Detect regime shifts early and adapt
5. Build a growing knowledge base of market patterns
"""
import json
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class RegimeRecord:
    """A record of what happened in a specific regime."""
    regime: str
    start_time: float
    end_time: float = 0.0
    duration_seconds: float = 0.0
    symbols_traded: List[str] = field(default_factory=list)
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    total_roi: float = 0.0
    best_symbol: str = ""
    worst_symbol: str = ""
    # What worked
    winning_strategies: List[str] = field(default_factory=list)
    winning_setups: List[Dict] = field(default_factory=list)
    # What didn't work
    losing_strategies: List[str] = field(default_factory=list)
    losing_setups: List[Dict] = field(default_factory=list)
    # Cumulative ROI per symbol within this regime record. Persisted so
    # best/worst ranking reflects the whole window, not just the last trade.
    symbol_roi_totals: Dict[str, float] = field(default_factory=dict)


@dataclass
class RegimeInsight:
    """An actionable insight from regime analysis."""
    regime: str
    insight_type: str  # "do", "avoid", "prefer", "warning"
    description: str
    confidence: float  # 0.0-1.0
    evidence: List[str] = field(default_factory=list)
    symbol: str = ""
    strategy: str = ""


class RegimeMemory:
    """Learns and remembers regime-specific trading patterns."""

    # Regime detection thresholds
    MIN_TRADES_FOR_INSIGHT = 3
    INSIGHT_CONFIDENCE_THRESHOLD = 0.6

    def __init__(self, db=None, redis_client=None):
        self.db = db
        self.redis = redis_client
        self.memory_file = "data/regime_memory.json"
        self.insights_file = "data/regime_insights.jsonl"
        self.records: Dict[str, RegimeRecord] = {}
        self.insights: List[RegimeInsight] = []
        self.current_regime: str = "UNKNOWN"
        self.regime_start_time: float = time.time()
        self._load_memory()
        # Migrate legacy records so prefer/avoid insights aren't all suppressed
        try:
            self.backfill_symbol_roi_totals()
        except Exception as e:
            log.debug(f"[REGIME_MEMORY] backfill skipped: {e}")

    def _load_memory(self):
        """Load regime memory from disk."""
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    data = json.load(f)
                for key, vals in data.get("records", {}).items():
                    # Older files lack symbol_roi_totals -> dataclass default fills it
                    self.records[key] = RegimeRecord(**vals)
                log.info(f"[REGIME_MEMORY] Loaded {len(self.records)} regime records")
            except Exception as e:
                log.error(f"[REGIME_MEMORY] Failed to load memory: {e}")

    def _save_memory(self):
        """Persist regime memory to disk."""
        os.makedirs(os.path.dirname(self.memory_file), exist_ok=True)
        data = {
            "records": {},
            "last_updated": datetime.utcnow().isoformat(),
        }
        for key, record in self.records.items():
            data["records"][key] = {
                "regime": record.regime,
                "start_time": record.start_time,
                "end_time": record.end_time,
                "duration_seconds": record.duration_seconds,
                "symbols_traded": record.symbols_traded,
                "total_trades": record.total_trades,
                "wins": record.wins,
                "losses": record.losses,
                "total_roi": record.total_roi,
                "best_symbol": record.best_symbol,
                "worst_symbol": record.worst_symbol,
                "winning_strategies": record.winning_strategies,
                "losing_strategies": record.losing_strategies,
                "symbol_roi_totals": record.symbol_roi_totals,
            }
        with open(self.memory_file, 'w') as f:
            json.dump(data, f, indent=2)

    def record_trade(self, symbol: str, regime: str, strategy: str,
                     roi: float, is_win: bool, market_context: Dict = None):
        """Record a trade outcome in the current regime."""
        # Update current regime if changed
        if regime != self.current_regime:
            self._end_regime()
            self.current_regime = regime
            self.regime_start_time = time.time()

        # Get or create regime record
        key = f"{regime}:{int(self.regime_start_time)}"
        if key not in self.records:
            self.records[key] = RegimeRecord(
                regime=regime,
                start_time=self.regime_start_time
            )

        record = self.records[key]
        record.total_trades += 1
        if is_win:
            record.wins += 1
        else:
            record.losses += 1
        record.total_roi += roi

        if symbol not in record.symbols_traded:
            record.symbols_traded.append(symbol)

        # Track strategies
        if is_win:
            if strategy not in record.winning_strategies:
                record.winning_strategies.append(strategy)
        else:
            if strategy not in record.losing_strategies:
                record.losing_strategies.append(strategy)

        # Track best/worst symbols from CUMULATIVE per-symbol ROI in this record.
        # (Previously symbol_rois was rebuilt from only the current trade each call,
        # so best_symbol == worst_symbol always, and _generate_insights emitted both
        # prefer AND avoid for the same symbol -- the HBAR/NEUTRAL contradiction.)
        if not isinstance(getattr(record, 'symbol_roi_totals', None), dict):
            record.symbol_roi_totals = {}
        record.symbol_roi_totals[symbol] = record.symbol_roi_totals.get(symbol, 0.0) + roi
        symbol_rois = dict(record.symbol_roi_totals)
        # Optional caller snapshot can extend the ranking universe (non-current symbols)
        if market_context and "symbol_rois" in market_context:
            for k, v in market_context["symbol_rois"].items():
                if k != symbol:
                    symbol_rois.setdefault(k, float(v))
        if symbol_rois:
            record.best_symbol = max(symbol_rois, key=symbol_rois.get)
            record.worst_symbol = min(symbol_rois, key=symbol_rois.get)

        # Save
        self._save_memory()

        # Generate insights if enough data
        if record.total_trades >= self.MIN_TRADES_FOR_INSIGHT:
            self._generate_insights(record)

        # Update Redis
        self._update_redis(record)

    def _end_regime(self):
        """End the current regime record."""
        key = f"{self.current_regime}:{int(self.regime_start_time)}"
        if key in self.records:
            record = self.records[key]
            record.end_time = time.time()
            record.duration_seconds = record.end_time - record.start_time
            self._save_memory()

    def _generate_insights(self, record: RegimeRecord):
        """Generate actionable insights from a regime record."""
        if record.total_trades < self.MIN_TRADES_FOR_INSIGHT:
            return

        win_rate = record.wins / record.total_trades
        avg_roi = record.total_roi / record.total_trades

        # Clear old insights for this regime
        self.insights = [i for i in self.insights if i.regime != record.regime]

        # Insight 1: Overall regime performance
        if win_rate > 0.60:
            self.insights.append(RegimeInsight(
                regime=record.regime,
                insight_type="do",
                description=f"{record.regime} regime is profitable (WR={win_rate:.0%}, avg ROI={avg_roi:.2f}%)",
                confidence=min(1.0, win_rate),
                evidence=[f"{record.total_trades} trades", f"{record.wins} wins"]
            ))
        elif win_rate < 0.35:
            self.insights.append(RegimeInsight(
                regime=record.regime,
                insight_type="warning",
                description=f"{record.regime} regime is losing (WR={win_rate:.0%}, avg ROI={avg_roi:.2f}%)",
                confidence=min(1.0, 1 - win_rate),
                evidence=[f"{record.total_trades} trades", f"{record.losses} losses"]
            ))

        # Insight 2: Best/worst symbols — only prefer symbols that made money,
        # only avoid symbols that lost money. When best == worst (single-symbol
        # window) exactly one of the two can fire, never both.
        roi_totals = getattr(record, 'symbol_roi_totals', None) or {}
        best_total = roi_totals.get(record.best_symbol, 0.0)
        worst_total = roi_totals.get(record.worst_symbol, 0.0)

        if record.best_symbol and best_total > 0:
            self.insights.append(RegimeInsight(
                regime=record.regime,
                insight_type="prefer",
                description=f"Trade {record.best_symbol} in {record.regime}",
                confidence=0.7,
                symbol=record.best_symbol,
                evidence=[f"Historical best in this regime (cum ROI {best_total:+.2f}%)"]
            ))

        if record.worst_symbol and worst_total < 0:
            self.insights.append(RegimeInsight(
                regime=record.regime,
                insight_type="avoid",
                description=f"Avoid {record.worst_symbol} in {record.regime}",
                confidence=0.7,
                symbol=record.worst_symbol,
                evidence=[f"Historical worst in this regime (cum ROI {worst_total:+.2f}%)"]
            ))

        # Insight 3: Strategy preferences
        if record.winning_strategies:
            for strat in record.winning_strategies[:2]:
                self.insights.append(RegimeInsight(
                    regime=record.regime,
                    insight_type="prefer",
                    description=f"Use {strat} strategy in {record.regime}",
                    confidence=0.65,
                    strategy=strat,
                    evidence=["Winning strategy in this regime"]
                ))

        if record.losing_strategies:
            for strat in record.losing_strategies[:2]:
                self.insights.append(RegimeInsight(
                    regime=record.regime,
                    insight_type="avoid",
                    description=f"Avoid {strat} strategy in {record.regime}",
                    confidence=0.65,
                    strategy=strat,
                    evidence=["Losing strategy in this regime"]
                ))

        # Store insights
        self._store_insights()

    def _store_insights(self):
        """Store insights to JSONL file."""
        os.makedirs(os.path.dirname(self.insights_file), exist_ok=True)
        with open(self.insights_file, 'w') as f:
            for insight in self.insights:
                f.write(json.dumps({
                    "timestamp": datetime.utcnow().isoformat(),
                    "regime": insight.regime,
                    "type": insight.insight_type,
                    "description": insight.description,
                    "confidence": insight.confidence,
                    "symbol": insight.symbol,
                    "strategy": insight.strategy,
                    "evidence": insight.evidence,
                }) + '\n')

    def _update_redis(self, record: RegimeRecord):
        """Update Redis with regime data."""
        if not self.redis:
            return

        key = f"regime_memory:{record.regime}:{int(record.start_time)}"
        self.redis.hset(key, mapping={
            "regime": record.regime,
            "total_trades": str(record.total_trades),
            "wins": str(record.wins),
            "losses": str(record.losses),
            "win_rate": f"{record.wins / record.total_trades:.4f}" if record.total_trades > 0 else "0",
            "total_roi": f"{record.total_roi:.4f}",
            "best_symbol": record.best_symbol,
            "worst_symbol": record.worst_symbol,
            "symbol_roi_totals": json.dumps(record.symbol_roi_totals or {}),
        })
        self.redis.expire(key, 86400 * 30)

    def get_recommendations(self, symbol: str, regime: str) -> List[RegimeInsight]:
        """Get trading recommendations for a symbol in a regime."""
        relevant = [i for i in self.insights if i.regime == regime]
        
        # Filter by symbol relevance
        symbol_relevant = [i for i in relevant if i.symbol == symbol or i.symbol == ""]
        
        # Sort by confidence
        return sorted(symbol_relevant, key=lambda i: i.confidence, reverse=True)

    def should_trade_symbol(self, symbol: str, regime: str) -> Tuple[bool, str]:
        """Check if a symbol should be traded in the current regime."""
        recommendations = self.get_recommendations(symbol, regime)
        
        # Check for explicit avoid warnings
        for rec in recommendations:
            if rec.insight_type == "avoid" and rec.symbol == symbol:
                if rec.confidence > self.INSIGHT_CONFIDENCE_THRESHOLD:
                    return False, f"Avoided: {rec.description}"
        
        # Check for positive preference
        for rec in recommendations:
            if rec.insight_type == "prefer" and rec.symbol == symbol:
                if rec.confidence > self.INSIGHT_CONFIDENCE_THRESHOLD:
                    return True, f"Preferred: {rec.description}"
        
        return True, "No strong signal"

    def get_regime_performance(self, regime: str) -> Dict:
        """Get historical performance for a regime."""
        regime_records = [r for r in self.records.values() if r.regime == regime]
        
        if not regime_records:
            return {"trades": 0, "win_rate": 0, "avg_roi": 0}
        
        total_trades = sum(r.total_trades for r in regime_records)
        total_wins = sum(r.wins for r in regime_records)
        total_roi = sum(r.total_roi for r in regime_records)
        
        return {
            "trades": total_trades,
            "win_rate": total_wins / total_trades if total_trades > 0 else 0,
            "avg_roi": total_roi / total_trades if total_trades > 0 else 0,
            "sessions": len(regime_records),
        }

    def get_memory_report(self) -> Dict:
        """Get comprehensive regime memory report."""
        report = {
            "current_regime": self.current_regime,
            "regime_start": datetime.fromtimestamp(self.regime_start_time).isoformat(),
            "total_records": len(self.records),
            "total_insights": len(self.insights),
            "regime_performance": {},
        }

        # Performance by regime
        regimes = set(r.regime for r in self.records.values())
        for regime in regimes:
            report["regime_performance"][regime] = self.get_regime_performance(regime)

        return report

    def backfill_symbol_roi_totals(self):
        """One-time migration: rebuild symbol_roi_totals for records loaded from
        older files that predate the field (all-zero totals would suppress every
        prefer/avoid insight). Uses total_roi as a floor proxy when per-symbol
        breakdown is unavailable — only fills empty maps for single-symbol records.
        """
        changed = False
        for record in self.records.values():
            if not getattr(record, "symbol_roi_totals", None):
                if len(record.symbols_traded) == 1:
                    record.symbol_roi_totals = {record.symbols_traded[0]: record.total_roi}
                    changed = True
                elif record.best_symbol or record.worst_symbol:
                    # Multi-symbol legacy record: seed both ends from total so
                    # ranking is at least sign-consistent until new trades accumulate
                    seed = {}
                    if record.best_symbol:
                        seed[record.best_symbol] = max(record.total_roi, 0.01)
                    if record.worst_symbol and record.worst_symbol not in seed:
                        seed[record.worst_symbol] = min(record.total_roi, -0.01)
                    if record.worst_symbol == record.best_symbol:
                        seed[record.best_symbol] = record.total_roi
                    record.symbol_roi_totals = seed
                    changed = True
        if changed:
            self._save_memory()
            log.info("[REGIME_MEMORY] Backfilled symbol_roi_totals for legacy records")
        return changed
