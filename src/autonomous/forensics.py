"""
Post-Trade Forensics: Analyzes WHY trades win or lose.

After each trade closes, this module:
1. Classifies the failure/success mode
2. Extracts contributing factors (regime, signal quality, timing, sizing)
3. Stores actionable lessons for future trade gating
4. Feeds insights back to learning_module and regime_memory
"""
import json
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict

log = logging.getLogger(__name__)

from src.intelligence.laya_client import (
    get_laya_client, LAYA_FORENSICS_QUESTIONS, extract_failure_mode,
)

@dataclass
class TradeForensics:
    """Complete forensic analysis of a closed trade."""
    trade_id: str
    symbol: str
    side: str  # LONG or SHORT
    entry_price: float
    exit_price: float
    roi_pct: float
    held_seconds: float
    regime_at_entry: str
    regime_at_exit: str
    meta_confidence: float
    quant_action: str
    # Derived analysis
    failure_mode: str  # TIMING, REGIME_MISMATCH, SIZING, SIGNAL_QUALITY, STOP_HUNT, GOOD_EXECUTION, UNKNOWN
    contributing_factors: List[str]
    lesson: str
    severity: str  # CATASTROPHIC (>10%), MAJOR (5-10%), MINOR (1-5%), NEGLIGIBLE (<1%)
    should_ban_symbol: bool
    confidence_adjustment: float  # 0.0-1.0 multiplier for future trades


class PostTradeForensics:
    """Analyzes closed trades and extracts actionable lessons."""

    # Failure mode thresholds
    CATASTROPHIC_LOSS = -10.0  # %
    MAJOR_LOSS = -5.0
    MINOR_LOSS = -1.0
    GOOD_PROFIT = 2.0
    TIMING_THRESHOLD_HOURS = 0.5  # Less than 30 min = timing issue
    STALE_THRESHOLD_HOURS = 8.0  # Held too long

    def __init__(self, db=None, redis_client=None, learning_module=None):
        self.db = db
        self.redis = redis_client
        self.learning_module = learning_module
        self.lessons_file = "data/trade_lessons.jsonl"
        self.opinions_file = "data/laya_opinions.jsonl"
        self.laya_client = get_laya_client()
        self._ensure_lessons_file()

    def _ensure_lessons_file(self):
        os.makedirs(os.path.dirname(self.lessons_file), exist_ok=True)
        if not os.path.exists(self.lessons_file):
            open(self.lessons_file, 'w').close()

    def analyze_trade(self, trade_data: Dict) -> TradeForensics:
        """
        Perform full forensic analysis on a closed trade.
        
        trade_data should contain:
        - trade_id, symbol, side, entry_price, exit_price
        - roi_pct, held_seconds
        - regime_at_entry, regime_at_exit
        - meta_confidence, quant_action
        - metadata (full JSONB from system_trades)
        """
        roi = trade_data.get('roi_pct', 0)
        held_hours = trade_data.get('held_seconds', 0) / 3600
        regime_entry = trade_data.get('regime_at_entry', 'UNKNOWN')
        regime_exit = trade_data.get('regime_at_exit', 'UNKNOWN')
        confidence = trade_data.get('meta_confidence', 0.5)

        # Classify severity
        severity = self._classify_severity(roi)

        # Classify failure mode
        failure_mode, factors = self._classify_failure_mode(trade_data)

        # LAYA second opinion (background, growth artifact): appends a
        # rules-vs-laya comparison to data/laya_opinions.jsonl for future
        # calibration of the classifier. Non-blocking; failures are silent.
        try:
            state = {"symbol": trade_data.get("symbol", ""),
                     "body": self._forensics_state(trade_data)}
            self.laya_client.evaluate(
                state,
                LAYA_FORENSICS_QUESTIONS,
                cache_key=f"forensic:{trade_data.get('trade_id', '')}:{time.time_ns()}",
                sink=lambda result, st, _td=trade_data, _fm=failure_mode:
                    self._append_laya_opinion(result, _td, _fm),
                store=False,
            )
        except Exception as e:
            log.debug(f"[LAYA] forensics opinion skipped: {e}")

        # LAYA CLOSED-LOOP LEARNING: send the realized outcome back as ground
        # truth for the ENTRY verdict Laya gave when this trade was opened.
        try:
            meta = trade_data.get("metadata") or {}
            if isinstance(meta, str):
                meta = json.loads(meta)
            laya_entry = (meta or {}).get("laya") or {}
            decision_id = laya_entry.get("decision_id")
            if decision_id:
                roi_pct = trade_data.get("roi_pct", 0)
                ground_truth = "enter" if roi_pct > 0 else "avoid"
                laya_choice = laya_entry.get("entry")
                reward = 1.0 if laya_choice == ground_truth else -1.0
                sent = self.laya_client.submit_feedback(
                    decision_id=decision_id,
                    question_id="entry",
                    ground_truth=ground_truth,
                    reward=reward,
                    target_type="choice",
                    notes=(f"{trade_data.get('symbol')} {trade_data.get('side')} "
                           f"roi={roi_pct:.2f}% mode={failure_mode} "
                           f"laya={laya_choice} agreed={reward > 0}"),
                )
                if sent:
                    log.info(f"[LAYA_FEEDBACK] entry verdict for {trade_data.get('symbol')} "
                             f"updated: truth={ground_truth} laya={laya_choice} "
                             f"reward={reward:+.0f} (roi={roi_pct:.2f}%)")
        except Exception as e:
            log.debug(f"[LAYA_FEEDBACK] entry feedback skipped: {e}")

        # Generate lesson
        lesson = self._generate_lesson(trade_data, failure_mode, factors)

        # Determine if symbol should be banned
        should_ban = self._should_ban(trade_data, roi, failure_mode)

        # Calculate confidence adjustment for future trades
        conf_adj = self._calculate_confidence_adjustment(trade_data, failure_mode)

        forensics = TradeForensics(
            trade_id=str(trade_data.get('trade_id', '')),
            symbol=trade_data.get('symbol', ''),
            side=trade_data.get('side', ''),
            entry_price=trade_data.get('entry_price', 0),
            exit_price=trade_data.get('exit_price', 0),
            roi_pct=roi,
            held_seconds=trade_data.get('held_seconds', 0),
            regime_at_entry=regime_entry,
            regime_at_exit=regime_exit,
            meta_confidence=confidence,
            quant_action=trade_data.get('quant_action', ''),
            failure_mode=failure_mode,
            contributing_factors=factors,
            lesson=lesson,
            severity=severity,
            should_ban_symbol=should_ban,
            confidence_adjustment=conf_adj,
        )

        # Store lesson
        self._store_lesson(forensics)

        # Update Redis for real-time access
        self._update_redis(forensics)

        # Feed to learning module if available
        if self.learning_module:
            self._feed_learning_module(forensics)

        log.info(f"[FORENSICS] {forensics.symbol} {forensics.side} ROI={forensics.roi_pct:.2f}% "
                 f"mode={forensics.failure_mode} severity={forensics.severity} "
                 f"lesson={forensics.lesson[:80]}")

        return forensics

    def _classify_severity(self, roi: float) -> str:
        if roi <= self.CATASTROPHIC_LOSS:
            return "CATASTROPHIC"
        elif roi <= self.MAJOR_LOSS:
            return "MAJOR"
        elif roi <= self.MINOR_LOSS:
            return "MINOR"
        elif roi >= self.GOOD_PROFIT:
            return "GOOD"
        else:
            return "NEGLIGIBLE"

    def _classify_failure_mode(self, trade_data: Dict) -> Tuple[str, List[str]]:
        """Classify why the trade won or lost. Returns (mode, factors)."""
        roi = trade_data.get('roi_pct', 0)
        held_hours = trade_data.get('held_seconds', 0) / 3600
        regime_entry = trade_data.get('regime_at_entry', 'UNKNOWN')
        regime_exit = trade_data.get('regime_at_exit', 'UNKNOWN')
        confidence = trade_data.get('meta_confidence', 0.5)
        quant_action = trade_data.get('quant_action', '')
        side = trade_data.get('side', '')
        factors = []

        # WINNER analysis
        if roi > 0:
            if held_hours < self.TIMING_THRESHOLD_HOURS:
                factors.append("quick_profit_taking")
            if regime_entry == regime_exit:
                factors.append("regime_consistent")
            if confidence > 0.7:
                factors.append("high_confidence_correct")
            return "GOOD_EXECUTION", factors

        # LOSER analysis
        # 1. Timing issues
        if held_hours < self.TIMING_THRESHOLD_HOURS:
            factors.append("entered_too_soon")
            return "TIMING", factors

        if held_hours > self.STALE_THRESHOLD_HOURS:
            factors.append("held_too_long_no_exit")
            return "STALE_POSITION", factors

        # 2. Regime mismatch
        if regime_entry != regime_exit and regime_entry != "UNKNOWN" and regime_exit != "UNKNOWN":
            factors.append(f"regime_shifted_{regime_entry}_to_{regime_exit}")
            return "REGIME_MISMATCH", factors

        # 3. Signal quality
        if confidence < 0.55:
            factors.append(f"low_confidence_{confidence:.2f}")
            return "SIGNAL_QUALITY", factors

        # 4. Large loss with high confidence = stop hunt or slippage
        if roi < self.MAJOR_LOSS and confidence > 0.65:
            factors.append("high_confidence_big_loss")
            return "STOP_HUNT", factors

        # 5. Default
        if roi < self.MINOR_LOSS:
            factors.append("general_inefficiency")
            return "SIZING", factors

        # 6. Rules inconclusive: let Laya's System-1 classifier supply the mode.
        # Blocking but short-timeout and fail-open -- service down keeps UNKNOWN.
        laya_mode = self._laya_classify_failure(trade_data)
        if laya_mode:
            return laya_mode

        return "UNKNOWN", factors

    @staticmethod
    def _forensics_state(trade_data: Dict) -> str:
        """Compact one-paragraph summary of a closed trade for laya."""
        try:
            meta = trade_data.get("metadata") or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    meta = {}
            laya_meta = meta.get("laya") if isinstance(meta, dict) else None
            if isinstance(laya_meta, dict):
                laya_desc = laya_meta.get("summary") or laya_meta.get("entry") or "NA"
            else:
                laya_desc = laya_meta or "NA"
            return (
                f"Trade {trade_data.get('symbol', '?')} {trade_data.get('side', '?')} "
                f"roi={trade_data.get('roi_pct', 0):.2f}% "
                f"held={trade_data.get('held_seconds', 0) / 3600:.2f}h "
                f"entry_regime={trade_data.get('regime_at_entry', 'UNKNOWN')} "
                f"exit_regime={trade_data.get('regime_at_exit', 'UNKNOWN')} "
                f"meta_confidence={trade_data.get('meta_confidence', 0):.2f} "
                f"quant_action={trade_data.get('quant_action', '?')} "
                f"laya_entry={laya_desc}"
            )
        except Exception:
            return f"{trade_data.get('symbol', '?')} closed with roi={trade_data.get('roi_pct', 0)}%"

    def _laya_classify_failure(self, trade_data: Dict) -> Optional[Tuple[str, List[str]]]:
        """Blocking laya failure-mode classification with fail-open semantics."""
        state = {"symbol": trade_data.get("symbol", ""),
                 "body": self._forensics_state(trade_data)}
        verdict = self.laya_client.evaluate_blocking(
            state, LAYA_FORENSICS_QUESTIONS, timeout=15.0)
        parsed = extract_failure_mode(verdict)
        if parsed and parsed["prob"] >= 0.50 and parsed["mode"] != "UNKNOWN":
            log.info(f"[LAYA_FORENSICS] {trade_data.get('symbol')} classified as "
                     f"{parsed['mode']} (p={parsed['prob']:.2f}) -- rules were inconclusive")
            return parsed["mode"], [f"laya_classified_p{parsed['prob']:.2f}"]
        return None

    def _append_laya_opinion(self, result: Dict, trade_data: Dict, rules_mode: str):
        """Worker-thread sink: append laya's second opinion next to the rules verdict."""
        parsed = extract_failure_mode(result)
        line = {
            "timestamp": datetime.utcnow().isoformat(),
            "trade_id": trade_data.get("trade_id", ""),
            "symbol": trade_data.get("symbol", ""),
            "side": trade_data.get("side", ""),
            "roi_pct": trade_data.get("roi_pct", 0),
            "held_hours": round(trade_data.get("held_seconds", 0) / 3600, 3),
            "regime_at_entry": trade_data.get("regime_at_entry", "UNKNOWN"),
            "meta_confidence": trade_data.get("meta_confidence", 0),
            "rules_mode": rules_mode,
            "laya_mode": parsed["mode"] if parsed else None,
            "laya_prob": round(parsed["prob"], 4) if parsed else None,
            "agree": bool(parsed) and parsed["mode"] == rules_mode,
        }
        os.makedirs(os.path.dirname(self.opinions_file), exist_ok=True)
        with open(self.opinions_file, "a") as f:
            f.write(json.dumps(line) + "\n")

        # Teacher feedback: when the rules produced a definitive mode, send it
        # as ground truth so laya's failure classifier calibrates to our data.
        if parsed and rules_mode and rules_mode != "UNKNOWN":
            decision_id = (result or {}).get("decision_id")
            if decision_id:
                reward = 1.0 if parsed["mode"] == rules_mode else -1.0
                self.laya_client.submit_feedback(
                    decision_id=decision_id,
                    question_id="failure_mode",
                    ground_truth=rules_mode,
                    reward=reward,
                    target_type="choice",
                    notes=(f"{trade_data.get('symbol')} roi={trade_data.get('roi_pct', 0):.2f}% "
                           f"laya={parsed['mode']} rules={rules_mode}"),
                )

    def _generate_lesson(self, trade_data: Dict, failure_mode: str, factors: List[str]) -> str:
        """Generate a human-readable lesson from the analysis."""
        symbol = trade_data.get('symbol', '')
        side = trade_data.get('side', '')
        roi = trade_data.get('roi_pct', 0)
        regime = trade_data.get('regime_at_entry', 'UNKNOWN')

        lessons = {
            "TIMING": f"{symbol} {side} entered too early in {regime} regime. "
                      f"Wait for confirmation before entry. Loss: {roi:.1f}%",
            
            "REGIME_MISMATCH": f"{symbol} {side} traded against regime shift. "
                               f"Regime changed from {regime} during hold. "
                               f"Use regime gates to prevent mismatched trades.",
            
            "SIGNAL_QUALITY": f"{symbol} {side} had weak signal (confidence={trade_data.get('meta_confidence', 0):.2f}). "
                              f"Increase min_confidence threshold for this symbol.",
            
            "STALE_POSITION": f"{symbol} {side} held too long without hitting TP/SL. "
                              f"Implement time-based exit for positions held >{self.STALE_THRESHOLD_HOURS}h.",
            
            "STOP_HUNT": f"{symbol} {side} hit stop despite high confidence. "
                         f"Possible stop hunt or oracle mismatch. Widen stops or use limit orders.",
            
            "SIZING": f"{symbol} {side} position size may be too large for the move. "
                      f"Reduce position size or tighten stop loss.",
            
            "GOOD_EXECUTION": f"{symbol} {side} profitable. "
                              f"Regime={regime}, confidence={trade_data.get('meta_confidence', 0):.2f}. "
                              f"Replicate this setup.",
            
            "UNKNOWN": f"{symbol} {side} closed with {roi:.1f}% ROI. "
                       f"Insufficient data to classify. Monitor similar setups.",
        }

        return lessons.get(failure_mode, f"{symbol} {side}: {roi:.1f}% ROI in {regime}")

    def _should_ban(self, trade_data: Dict, roi: float, failure_mode: str) -> bool:
        """Determine if the symbol should be temporarily banned (Disabled: user constraint strictly prohibits pair blacklisting)."""
        return False


    def _calculate_confidence_adjustment(self, trade_data: Dict, failure_mode: str) -> float:
        """Calculate how much to adjust confidence for future trades on this symbol."""
        roi = trade_data.get('roi_pct', 0)

        # Catastrophic loss: heavily penalize
        if roi <= self.CATASTROPHIC_LOSS:
            return 0.3

        # Major loss: significant penalty
        if failure_mode == "REGIME_MISMATCH":
            return 0.6  # Don't trade this regime for this symbol
        elif failure_mode == "SIGNAL_QUALITY":
            return 0.7  # Need stronger signals
        elif failure_mode == "TIMING":
            return 0.8  # Wait for confirmation
        elif failure_mode == "STALE_POSITION":
            return 0.85  # Tighter time stops
        elif failure_mode == "STOP_HUNT":
            return 0.75  # Wider stops needed
        elif failure_mode == "GOOD_EXECUTION":
            return 1.1  # Boost confidence (capped at 1.0 later)

        return 0.9  # Default slight penalty for any loss

    def _store_lesson(self, forensics: TradeForensics):
        """Store lesson to JSONL file for persistence."""
        lesson_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "trade_id": forensics.trade_id,
            "symbol": forensics.symbol,
            "side": forensics.side,
            "roi_pct": forensics.roi_pct,
            "held_seconds": forensics.held_seconds,
            "regime_at_entry": forensics.regime_at_entry,
            "failure_mode": forensics.failure_mode,
            "contributing_factors": forensics.contributing_factors,
            "lesson": forensics.lesson,
            "severity": forensics.severity,
            "confidence_adjustment": forensics.confidence_adjustment,
            "should_ban": forensics.should_ban_symbol,
        }

        with open(self.lessons_file, 'a') as f:
            f.write(json.dumps(lesson_data) + '\n')

    def _update_redis(self, forensics: TradeForensics):
        """Update Redis with real-time forensic data."""
        if not self.redis:
            return

        # Per-symbol forensics
        key = f"forensics:{forensics.symbol}"
        self.redis.hset(key, mapping={
            "last_failure_mode": forensics.failure_mode,
            "last_roi": str(forensics.roi_pct),
            "last_severity": forensics.severity,
            "confidence_adj": str(forensics.confidence_adjustment),
            "should_ban": str(forensics.should_ban_symbol),
            "last_analyzed": str(int(time.time())),
        })
        self.redis.expire(key, 86400 * 7)  # 7 day TTL

        # Global failure mode counts
        mode_key = f"failure_modes:{forensics.failure_mode}"
        self.redis.incr(mode_key)
        self.redis.expire(mode_key, 86400 * 30)

        # Regime performance tracking
        if forensics.regime_at_entry != "UNKNOWN":
            regime_key = f"regime_perf:{forensics.regime_at_entry}"
            if forensics.roi_pct > 0:
                self.redis.hincrby(regime_key, "wins", 1)
            else:
                self.redis.hincrby(regime_key, "losses", 1)
            self.redis.hincrbyfloat(regime_key, "total_roi", forensics.roi_pct)
            self.redis.expire(regime_key, 86400 * 30)

    def _feed_learning_module(self, forensics: TradeForensics):
        """Feed forensic insights back to the learning module."""
        if not self.learning_module:
            return

        # Update symbol confidence adjustment
        symbol = forensics.symbol
        current_adj = self.learning_module.symbol_stats.get(symbol, {}).get('confidence_adjustment', 1.0)
        # Blend: 70% old, 30% new (EMA-like)
        new_adj = (0.7 * current_adj) + (0.3 * forensics.confidence_adjustment)
        new_adj = max(0.3, min(1.2, new_adj))  # Clamp

        if symbol not in self.learning_module.symbol_stats:
            self.learning_module.symbol_stats[symbol] = {'wins': 0, 'losses': 0, 'total': 0, 'total_roi': 0}
        self.learning_module.symbol_stats[symbol]['confidence_adjustment'] = new_adj

    def get_symbol_health(self, symbol: str) -> Dict:
        """Get current health status for a symbol based on forensic history."""
        if not self.redis:
            return {"status": "unknown"}

        key = f"forensics:{symbol}"
        data = self.redis.hgetall(key)

        if not data:
            return {"status": "no_data"}

        return {
            "status": "active",
            "last_failure_mode": data.get("last_failure_mode", "UNKNOWN"),
            "last_roi": float(data.get("last_roi", 0)),
            "severity": data.get("last_severity", "UNKNOWN"),
            "confidence_adjustment": float(data.get("confidence_adj", 1.0)),
            "last_analyzed": int(data.get("last_analyzed", 0)),
        }

    def get_all_lessons(self, limit: int = 50) -> List[Dict]:
        """Get recent lessons from the lessons file."""
        lessons = []
        if not os.path.exists(self.lessons_file):
            return lessons

        with open(self.lessons_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        lessons.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        return lessons[-limit:]

    def get_failure_mode_stats(self) -> Dict:
        """Get global failure mode distribution."""
        if not self.redis:
            return {}

        modes = ["TIMING", "REGIME_MISMATCH", "SIGNAL_QUALITY", 
                 "STALE_POSITION", "STOP_HUNT", "SIZING", "GOOD_EXECUTION", "UNKNOWN"]
        stats = {}
        for mode in modes:
            key = f"failure_modes:{mode}"
            count = self.redis.get(key)
            stats[mode] = int(count) if count else 0

        return stats
