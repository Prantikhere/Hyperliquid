"""
Learning Module: Tracks past failures and prevents similar behavior.
This module ensures the system learns from past mistakes and never repeats them.
"""

import json
import os
from datetime import datetime
from src.utils.logger import log


class LearningModule:
    """Tracks trade outcomes and learns from failures to prevent repetition."""
    
    def __init__(self):
        self.learning_file = "data/learning_outcomes.jsonl"
        self.failure_patterns = self._load_failure_patterns()
    
    def _load_failure_patterns(self):
        """Load known failure patterns from learning file."""
        patterns = {
            "large_losses": [],  # Trades with >5% loss
            "repeated_failures": [],  # Same symbol failing multiple times
            "regime_mismatches": [],  # Trades in wrong regime
            "timing_failures": [],  # Trades with bad entry timing
        }
        
        if os.path.exists(self.learning_file):
            try:
                with open(self.learning_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            if data.get('outcome_roi', 0) < -0.05:  # >5% loss
                                patterns["large_losses"].append(data)
                            if data.get('repeat_failure', False):
                                patterns["repeated_failures"].append(data)
            except Exception as e:
                log.error(f"Error loading learning patterns: {e}")
        
        return patterns
    
    def record_trade_outcome(self, symbol, exchange, side, entry_price, exit_price, 
                            roi, regime, comp_score, holding_time_hours):
        """Record trade outcome for learning."""
        outcome = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "exchange": exchange,
            "side": side,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "outcome_roi": roi,
            "regime": regime,
            "composite_score": comp_score,
            "holding_time_hours": holding_time_hours,
            "is_failure": roi < -0.03,  # >3% loss = failure
            "failure_reason": self._identify_failure_reason(roi, regime, comp_score, holding_time_hours)
        }
        
        # Append to learning file
        try:
            with open(self.learning_file, 'a') as f:
                f.write(json.dumps(outcome) + '\n')
        except Exception as e:
            log.error(f"Error recording trade outcome: {e}")
        
        return outcome
    
    def _identify_failure_reason(self, roi, regime, comp_score, holding_time):
        """Identify why a trade failed."""
        if roi < -0.05:
            return "LARGE_LOSS"
        if roi < -0.03 and holding_time > 4:
            return "STALE_POSITION"
        if roi < -0.03 and comp_score < 0.3:
            return "WEAK_SIGNAL"
        if "HIGH_VOL" in str(regime) and roi < -0.02:
            return "REGIME_MISMATCH"
        return "NORMAL"
    
    def should_trade_symbol(self, symbol, exchange):
        """Check if we should trade this symbol based on past failures."""
        # Check for repeated failures
        repeated_failures = [
            f for f in self.failure_patterns["repeated_failures"]
            if f.get("symbol") == symbol and f.get("exchange") == exchange
        ]
        
        if len(repeated_failures) >= 3:
            log.warning(f"AVOIDING {symbol} on {exchange}: {len(repeated_failures)} past failures")
            return False
        
        # Check for recent large losses
        recent_large_losses = [
            f for f in self.failure_patterns["large_losses"]
            if f.get("symbol") == symbol and f.get("exchange") == exchange
        ]
        
        if len(recent_large_losses) >= 2:
            log.warning(f"CAUTION: {symbol} on {exchange}: {len(recent_large_losses)} large losses")
            return False
        
        return True
    
    def get_adjusted_parameters(self, symbol, exchange, regime):
        """Get adjusted parameters based on past learnings."""
        adjustments = {
            "stop_loss_multiplier": 1.0,
            "take_profit_multiplier": 1.0,
            "position_size_multiplier": 1.0
        }
        
        # Check for regime mismatches
        regime_mismatches = [
            f for f in self.failure_patterns["regime_mismatches"]
            if f.get("symbol") == symbol
        ]
        
        if regime_mismatches:
            # Tighten stops in known bad regimes
            adjustments["stop_loss_multiplier"] = 0.8  # 20% tighter
            adjustments["position_size_multiplier"] = 0.7  # 30% smaller
        
        # Check for large losses
        large_losses = [
            f for f in self.failure_patterns["large_losses"]
            if f.get("symbol") == symbol
        ]
        
        if large_losses:
            # Be more conservative with symbols that had large losses
            adjustments["stop_loss_multiplier"] = 0.7  # 30% tighter
            adjustments["take_profit_multiplier"] = 1.2  # 20% wider targets
            adjustments["position_size_multiplier"] = 0.5  # 50% smaller
        
        return adjustments
    
    def get_learning_report(self):
        """Generate a learning report."""
        report = {
            "total_trades": 0,
            "failures": 0,
            "large_losses": 0,
            "repeated_failures": 0,
            "top_failure_reasons": {},
            "recommendations": []
        }
        
        if os.path.exists(self.learning_file):
            try:
                with open(self.learning_file, 'r') as f:
                    trades = [json.loads(line) for line in f if line.strip()]
                    report["total_trades"] = len(trades)
                    report["failures"] = len([t for t in trades if t.get("is_failure")])
                    report["large_losses"] = len([
                        t for t in trades if t.get("outcome_roi", 0) < -0.05
                    ])
                    
                    # Count failure reasons
                    for trade in trades:
                        reason = trade.get("failure_reason", "UNKNOWN")
                        report["top_failure_reasons"][reason] = report["top_failure_reasons"].get(reason, 0) + 1
                    
                    # Generate recommendations
                    if report["large_losses"] > 0:
                        report["recommendations"].append("Tighten stop losses further")
                    if report["failures"] > report["total_trades"] * 0.4:
                        report["recommendations"].append("Review entry signals")
                    if report["repeated_failures"] > 3:
                        report["recommendations"].append("Avoid problematic symbols")
                        
            except Exception as e:
                log.error(f"Error generating learning report: {e}")
        
        return report


# Global instance
learning_module = LearningModule()
