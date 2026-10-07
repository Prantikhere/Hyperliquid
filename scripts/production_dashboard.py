#!/usr/bin/env python3
"""Production monitoring dashboard for TradingBingx system.

Usage: python3 scripts/production_dashboard.py

Displays real-time system status, P&L, drawdown, and alerts.
"""

import os
import sys

# Auto-reexec with project venv python if invoked via system python
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VENV_DIR = os.path.join(_PROJECT_ROOT, "venv")
_VENV_PYTHON = os.path.join(_VENV_DIR, "bin", "python3")
if os.path.exists(_VENV_PYTHON) and sys.prefix != _VENV_DIR:
    os.execv(_VENV_PYTHON, [_VENV_PYTHON] + sys.argv)


import json
import redis
import time
from datetime import datetime

def main():
    # Connect to Redis
    try:
        r = redis.Redis(host=os.getenv('REDIS_HOST', 'localhost'), port=6379, decode_responses=True)
        r.ping()
    except Exception as e:
        print(f"ERROR: Cannot connect to Redis: {e}")
        return

    # Get account values
    peak_value = float(r.get("account_peak_value") or 0)
    current_value = float(r.get("account_current_value") or 0)
    
    # Calculate drawdown
    drawdown_pct = 0
    if peak_value > 0:
        drawdown_pct = ((peak_value - current_value) / peak_value) * 100

    # Get open positions count
    open_positions = int(r.get("open_positions_count") or 0)

    # Get session P&L
    session_key = f"session_realized_pnl:{int(time.time() / 86400)}"
    session_pnl = float(r.get(session_key) or 0)

    # Get churn guard counts
    churn_blocks = 0
    for key in r.scan_iter("entry_cooldown:*"):
        churn_blocks += 1

    # Get exit counts
    exit_counts = {}
    for key in r.scan_iter("exit_count:*"):
        parts = key.split(":")
        if len(parts) >= 3:
            symbol = parts[1]
            count = int(r.get(key) or 0)
            exit_counts[symbol] = count

    # Display dashboard
    print("=" * 60)
    print("TRADINGBINGX PRODUCTION DASHBOARD")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("ACCOUNT STATUS:")
    print(f"  Current Value: ${current_value:.2f}")
    print(f"  Peak Value:    ${peak_value:.2f}")
    print(f"  Drawdown:      {drawdown_pct:.1f}%")
    print(f"  Session P&L:   ${session_pnl:+.2f}")
    print()
    print("RISK STATUS:")
    print(f"  Open Positions: {open_positions} / 4")
    print(f"  Churn Blocks:   {churn_blocks}")
    print()
    print("EXIT COUNTS (today):")
    for symbol, count in sorted(exit_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {symbol}: {count}")
    print()
    
    # Alerts
    print("ALERTS:")
    if drawdown_pct >= 15:
        print(f"  CRITICAL: Max drawdown breached ({drawdown_pct:.1f}% >= 15%)")
    elif drawdown_pct >= 10:
        print(f"  WARNING: High drawdown ({drawdown_pct:.1f}% >= 10%)")
    else:
        print(f"  OK: Drawdown within limits ({drawdown_pct:.1f}%)")
    
    if open_positions >= 4:
        print(f"  WARNING: Max positions reached ({open_positions}/4)")
    else:
        print(f"  OK: Position count normal ({open_positions}/4)")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
