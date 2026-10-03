"""
Learning Module: Tracks past failures and prevents similar behavior.
This module ensures the system learns from past mistakes and never repeats them.
Capital Protection Mode: From here on, only making money and growth.
"""

import json
import os
from datetime import datetime
from src.utils.logger import log


class LearningModule:
    """Tracks trade outcomes and learns from failures to prevent repetition."""
    
    def __init__(self):
        self.learning_file = "data/learning_outcomes.jsonl"
        self.calibration_file = "data/calibration_ledger.jsonl"
        self.banned_symbols = set()  # Symbols with catastrophic losses
        self.symbol_stats = {}  # Per-symbol win/loss stats
        self.failure_patterns = self._load_failure_patterns()
        self.historical_data = self._load_historical_data()
    
    def _load_historical_data(self):
        """Load historical trade data from calibration ledger for learning."""
        data = []
        if os.path.exists(self.calibration_file):
            try:
                with open(self.calibration_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            trade = json.loads(line)
                            data.append(trade)
                            # Track per-symbol stats
                            sym = trade.get('symbol', '?')
                            roi = trade.get('outcome_roi', 0)
                            if sym not in self.symbol_stats:
                                self.symbol_stats[sym] = {'wins': 0, 'losses': 0, 'total': 0, 'total_roi': 0}
                            self.symbol_stats[sym]['total'] += 1
                            self.symbol_stats[sym]['total_roi'] += roi
                            if roi > 0:
                                self.symbol_stats[sym]['wins'] += 1
                            else:
                                self.symbol_stats[sym]['losses'] += 1
                            # Ban symbols with catastrophic losses
                            if roi < -0.10:  # >10% loss
                                self.banned_symbols.add(sym)
                                log.warning(f"[LEARNING] BANNED SYMBOL: {sym} (catastrophic loss: {roi*100:.1f}%)")
            except Exception as e:
                log.error(f"Error loading historical data: {e}")
        return data
    
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
        # CRITICAL: Check if symbol is banned (catastrophic loss >10%)
        if symbol in self.banned_symbols:
            log.warning(f"[CAPITAL_PROTECTION] BANNED: {symbol} - catastrophic loss in history")
            return False
        
        # Check per-symbol stats
        if symbol in self.symbol_stats:
            stats = self.symbol_stats[symbol]
            total = stats.get('total', 0)
            win_rate = stats.get('wins', 0) / total if total > 0 else 0
            avg_roi = stats.get('total_roi', 0.0) / total if total > 0 else 0
            
            # Ban symbols with <40% win rate and negative average ROI
            if win_rate < 0.40 and avg_roi < -0.02:
                log.warning(f"[CAPITAL_PROTECTION] AVOIDING {symbol}: win_rate={win_rate:.1%}, avg_roi={avg_roi:.2%}")
                return False
        
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
    
    def get_symbol_confidence_adjustment(self, symbol):
        """Get confidence adjustment based on historical performance."""
        if symbol not in self.symbol_stats:
            return 1.0  # No data, use default
        
        stats = self.symbol_stats[symbol]
        total = stats.get('total', 0)
        win_rate = stats.get('wins', 0) / total if total > 0 else 0
        avg_roi = stats.get('total_roi', 0.0) / total if total > 0 else 0
        
        # Reduce confidence for symbols with poor history
        if win_rate < 0.50 and avg_roi < 0:
            return 0.7  # 30% reduction
        elif win_rate < 0.60:
            return 0.85  # 15% reduction
        
        return 1.0  # No adjustment
    
    def get_adjusted_parameters(self, symbol, exchange, regime):
        """Get adjusted parameters based on past learnings."""
        adjustments = {
            "stop_loss_multiplier": 1.0,
            "take_profit_multiplier": 1.0,
            "position_size_multiplier": 1.0,
            "max_loss_pct": 0.02,  # Default 2% max loss
            "take_profit_pct": 0.05,  # Default 5% take profit
        }
        
        # Check per-symbol stats
        if symbol in self.symbol_stats:
            stats = self.symbol_stats[symbol]
            total = stats.get('total', 0)
            win_rate = stats.get('wins', 0) / total if total > 0 else 0
            avg_roi = stats.get('total_roi', 0.0) / total if total > 0 else 0
            
            # Aggressive risk management for symbols with poor history
            if win_rate < 0.50 and avg_roi < -0.02:
                adjustments["stop_loss_multiplier"] = 0.5  # 50% tighter stops
                adjustments["take_profit_multiplier"] = 0.8  # 20% wider targets
                adjustments["position_size_multiplier"] = 0.4  # 60% smaller positions
                adjustments["max_loss_pct"] = 0.01  # 1% max loss
                adjustments["take_profit_pct"] = 0.03  # 3% take profit
            elif win_rate < 0.60:
                adjustments["stop_loss_multiplier"] = 0.7  # 30% tighter stops
                adjustments["position_size_multiplier"] = 0.6  # 40% smaller positions
                adjustments["max_loss_pct"] = 0.015  # 1.5% max loss
        
        # Check for regime mismatches
        regime_mismatches = [
            f for f in self.failure_patterns["regime_mismatches"]
            if f.get("symbol") == symbol
        ]
        
        if regime_mismatches:
            adjustments["stop_loss_multiplier"] *= 0.8  # Additional 20% tighter
            adjustments["position_size_multiplier"] *= 0.7  # Additional 30% smaller
        
        # Check for large losses
        large_losses = [
            f for f in self.failure_patterns["large_losses"]
            if f.get("symbol") == symbol
        ]
        
        if large_losses:
            adjustments["stop_loss_multiplier"] *= 0.7  # Additional 30% tighter
            adjustments["position_size_multiplier"] *= 0.5  # Additional 50% smaller
            adjustments["max_loss_pct"] = 0.01  # 1% max loss for these symbols
        
        return adjustments
    
    def get_learning_report(self):
        """Generate a learning report."""
        report = {
            "total_trades": 0,
            "failures": 0,
            "large_losses": 0,
            "repeated_failures": 0,
            "top_failure_reasons": {},
            "recommendations": [],
            "banned_symbols": list(self.banned_symbols),
            "symbol_stats": {}
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
        
        # Add historical stats
        for sym, stats in self.symbol_stats.items():
            total = stats.get('total', 0)
            win_rate = stats.get('wins', 0) / total if total > 0 else 0
            avg_roi = stats.get('total_roi', 0.0) / total if total > 0 else 0
            report["symbol_stats"][sym] = {
                "trades": total,
                "wins": stats.get('wins', 0),
                "losses": stats.get('losses', 0),
                "win_rate": win_rate,
                "avg_roi": avg_roi
            }
        
        return report
    
    def check_capital_protection(self, symbol, confidence, regime, composite_score):
        """Check if trade meets capital protection requirements."""
        reasons = []
        
        # Check if symbol is banned
        if symbol in self.banned_symbols:
            reasons.append(f"Symbol {symbol} is banned (catastrophic loss history)")
        
        # Check confidence requirement
        if confidence < 0.40:
            reasons.append(f"Confidence too low: {confidence:.2f} (min: 0.40)")
        
        # Check regime requirement
        if regime is None or regime == "UNKNOWN":
            reasons.append(f"Regime unknown: {regime}")
        
        # Check composite score requirement
        if composite_score < 0.55 and composite_score > 0.45:
            reasons.append(f"Composite score in neutral zone: {composite_score:.2f}")
        
        # Check per-symbol stats
        if symbol in self.symbol_stats:
            stats = self.symbol_stats[symbol]
            total = stats.get('total', 0)
            win_rate = stats.get('wins', 0) / total if total > 0 else 0
            avg_roi = stats.get('total_roi', 0.0) / total if total > 0 else 0
            
            if win_rate < 0.40 and avg_roi < -0.02:
                reasons.append(f"Poor historical performance: win_rate={win_rate:.1%}, avg_roi={avg_roi:.2%}")
        
        if reasons:
            log.warning(f"[CAPITAL_PROTECTION] Trade blocked for {symbol}: {'; '.join(reasons)}")
            return False, reasons
        
        return True, []


# Global instance
learning_module = LearningModule()
