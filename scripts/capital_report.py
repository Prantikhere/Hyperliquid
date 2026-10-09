#!/usr/bin/env python3
"""
capital_report.py - Terminal & CLI capital and growth/loss reporter.
Fetches real-time equity, free margin, margin utilization, unrealized/realized P&L,
and open position states from Hyperliquid and Redis.
"""
import os
import sys
from pathlib import Path

# Auto re-exec in project virtualenv if needed
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_VENV_DIR = _PROJECT_ROOT / "venv"
_VENV_PYTHON = _VENV_DIR / "bin" / "python3"

if _VENV_PYTHON.exists() and sys.prefix != str(_VENV_DIR):
    os.execv(str(_VENV_PYTHON), [str(_VENV_PYTHON)] + sys.argv)

sys.path.insert(0, str(_PROJECT_ROOT))

import time
import json
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

from src.execution.hl_raw import HlSdkClient
import redis

def generate_report():
    r = redis.Redis(
        host=os.getenv("REDIS_HOST", "localhost"),
        port=int(os.getenv("REDIS_PORT", 6379)),
        decode_responses=True
    )

    hl = None
    try:
        hl = HlSdkClient()
        bal = hl.get_balance()
        raw_positions = hl.get_positions()
    except Exception as e:
        bal = {}
        raw_positions = []
        print(f"[ERROR] Failed to fetch Hyperliquid data: {e}", file=sys.stderr)
    finally:
        if hl:
            try:
                hl.close()
            except Exception:
                pass

    total_wallet = float(bal.get("account_value", 0.0))
    free_margin = float(bal.get("available", 0.0))
    margin_used = float(bal.get("margin_used", 0.0))
    margin_util_pct = (margin_used / total_wallet * 100.0) if total_wallet > 0 else 0.0

    # Redis tracking
    peak_val_str = r.get("account_peak_value")
    peak_val = float(peak_val_str) if peak_val_str else total_wallet

    today_key = f"session_realized_pnl_{int(time.time() // 86400)}"
    today_pnl_str = r.get(today_key)
    today_pnl = float(today_pnl_str) if today_pnl_str else 0.0

    # Drawdown from peak
    dd_usd = max(0.0, peak_val - total_wallet)
    dd_pct = (dd_usd / peak_val * 100.0) if peak_val > 0 else 0.0

    # IST timestamp (+5:30)
    ist = timezone(timedelta(hours=5, minutes=30))
    now_ist = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S IST")

    # Filter active positions where szi != 0
    active_positions = []
    unrealized_total = 0.0
    for item in raw_positions:
        p = item.get("position", {}) if isinstance(item, dict) else {}
        sz = float(p.get("szi", 0.0))
        if abs(sz) > 1e-6:
            upnl = float(p.get("unrealizedPnl", 0.0))
            unrealized_total += upnl
            active_positions.append({
                "coin": p.get("coin"),
                "size": sz,
                "entry_price": float(p.get("entryPx", 0.0)),
                "unrealized_pnl": upnl,
                "return_on_equity": float(p.get("returnOnEquity", 0.0)) * 100.0,
                "side": "LONG" if sz > 0 else "SHORT"
            })

    report_lines = []
    report_lines.append("=" * 64)
    report_lines.append(f" 📊 HYPERLIQUID WORKING CAPITAL & PERFORMANCE REPORT")
    report_lines.append(f" ⏰ As of: {now_ist}")
    report_lines.append("=" * 64)
    report_lines.append(f" Working Capital (Equity):  ${total_wallet:,.2f}")
    report_lines.append(f" Available Free Margin:     ${free_margin:,.2f}")
    report_lines.append(f" Margin In Use:             ${margin_used:,.2f} ({margin_util_pct:.1f}%)")
    report_lines.append(f" All-Time Peak Capital:     ${peak_val:,.2f}")
    report_lines.append(f" Drawdown from Peak:        -${dd_usd:,.2f} (-{dd_pct:.2f}%)")
    pnl_sign = "+" if today_pnl >= 0 else ""
    report_lines.append(f" Today Realized P&L:        {pnl_sign}${today_pnl:,.2f}")
    unr_sign = "+" if unrealized_total >= 0 else ""
    report_lines.append(f" Total Unrealized P&L:      {unr_sign}${unrealized_total:,.2f}")
    report_lines.append("-" * 64)
    report_lines.append(f" Open Positions Count:      {len(active_positions)} (Unrestricted)")

    if active_positions:
        report_lines.append("")
        report_lines.append(" Active Positions:")
        for idx, pos in enumerate(active_positions, 1):
            coin = pos["coin"]
            side = pos["side"]
            size = pos["size"]
            entry = pos["entry_price"]
            upnl = pos["unrealized_pnl"]
            roe = pos["return_on_equity"]
            u_sign = "+" if upnl >= 0 else ""
            r_sign = "+" if roe >= 0 else ""
            report_lines.append(
                f"   {idx}. {coin}/USDT [{side}] size={size} entry=${entry:.4f} | "
                f"uPnL: {u_sign}${upnl:.2f} (ROE: {r_sign}{roe:.2f}%)"
            )
    else:
        report_lines.append(" Status: All positions flat. Waiting for high-confluence entry.")

    report_lines.append("=" * 64)
    output = "\n".join(report_lines)
    return output

if __name__ == "__main__":
    report_text = generate_report()
    print(report_text)
