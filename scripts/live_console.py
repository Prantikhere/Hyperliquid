#!/usr/bin/env python3
"""TradingBingx Live Console Dashboard — Real-time terminal surveillance.

Usage:
  python3 scripts/live_console.py [--interval SECONDS]
"""
import os
import sys

# Auto-reexec with project venv python if invoked via system python
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VENV_DIR = os.path.join(_PROJECT_ROOT, "venv")
_VENV_PYTHON = os.path.join(_VENV_DIR, "bin", "python3")
if os.path.exists(_VENV_PYTHON) and sys.prefix != _VENV_DIR:
    os.execv(_VENV_PYTHON, [_VENV_PYTHON] + sys.argv)


import time
import argparse
from datetime import datetime

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

import redis
import requests
from src.execution.hl_raw import HlSdkClient

# ANSI colors
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
MAGENTA = "\033[95m"
CLEAR_SCREEN = "\033[2J\033[H"


def check_process(name: str) -> bool:
    import subprocess
    res = subprocess.run(["pgrep", "-a", "-f", name], capture_output=True, text=True)
    return res.returncode == 0


def format_currency(val: float) -> str:
    color = GREEN if val >= 0 else RED
    return f"{color}${val:+.2f}{RESET}"


def run_dashboard(interval: int = 5):
    client = HlSdkClient()
    r = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, decode_responses=True)

    while True:
        try:
            # 1. Fetch Account State
            state = client.info.user_state(client.wallet_address)
            margin = state.get("marginSummary", {})
            equity = float(margin.get("accountValue", 0))
            margin_used = float(margin.get("totalMarginUsed", 0))
            free_margin = max(0.0, equity - margin_used)
            margin_pct = (margin_used / equity * 100) if equity > 0 else 0
            safety_buffer = equity - 185.00

            positions = [p for p in state.get("assetPositions", []) if float(p["position"]["szi"]) != 0]

            # 2. Fetch Laya Health
            laya_status = "DOWN"
            laya_decisions = 0
            laya_feedback = 0
            try:
                resp = requests.get("http://127.0.0.1:8080/health", timeout=2)
                if resp.status_code == 200:
                    d = resp.json()
                    laya_status = f"UP (v{d.get('laya_version', '?')})"
                    laya_decisions = d.get("total_historical_decisions", 0)
                    laya_feedback = d.get("total_feedback_records", 0)
            except Exception:
                pass

            # 3. Process Status
            hl_ok = check_process("hl_executor")
            perp_ok = check_process("hl_perp_ls")
            pairs_ok = check_process("pairs_arb")
            watchdog_ok = check_process("watchdog.sh")

            # 4. Render Console
            output = []
            output.append(CLEAR_SCREEN)
            output.append(f"{BOLD}{CYAN}╔══════════════════════════════════════════════════════════════════════════════════════╗{RESET}")
            output.append(f"{BOLD}{CYAN}║                    TRADINGBINGX REAL-TIME SURVEILLANCE CONSOLE                       ║{RESET}")
            output.append(f"{BOLD}{CYAN}║                    {datetime.now().strftime('%Y-%m-%d %H:%M:%S IST')} | Floor: $185.00 | Mode: AGGRESSIVE_SCALP      ║{RESET}")
            output.append(f"{BOLD}{CYAN}╠══════════════════════════════════════════════════════════════════════════════════════╣{RESET}")

            # Capital Section
            eq_color = GREEN if equity >= 185.0 else RED
            buf_color = GREEN if safety_buffer >= 0 else RED
            output.append(f"  {BOLD}Total Equity:{RESET}  {eq_color}${equity:.2f}{RESET}  (Buffer above $185: {buf_color}+${safety_buffer:.2f}{RESET})")
            output.append(f"  {BOLD}Margin Used:{RESET}   ${margin_used:.2f} ({margin_pct:.1f}% util) | {BOLD}Free Margin:{RESET} ${free_margin:.2f}")

            # Margin Bar
            bar_len = 30
            filled = int((min(100.0, margin_pct) / 100.0) * bar_len)
            bar_color = GREEN if margin_pct < 75 else (YELLOW if margin_pct < 80 else RED)
            bar_str = bar_color + "█" * filled + "░" * (bar_len - filled) + RESET
            output.append(f"  {BOLD}Margin Bar:{RESET}    [{bar_str}] {margin_pct:.1f}% / 80.0% cap")
            output.append("")

            # Daemon Health
            p_hl = f"{GREEN}● RUNNING{RESET}" if hl_ok else f"{RED}✖ DOWN{RESET}"
            p_perp = f"{GREEN}● RUNNING{RESET}" if perp_ok else f"{YELLOW}✖ DOWN{RESET}"
            p_pairs = f"{GREEN}● RUNNING{RESET}" if pairs_ok else f"{YELLOW}✖ DOWN{RESET}"
            p_wd = f"{GREEN}● RUNNING{RESET}" if watchdog_ok else f"{YELLOW}✖ DOWN{RESET}"
            p_laya = f"{GREEN}● {laya_status}{RESET}" if "UP" in laya_status else f"{RED}✖ {laya_status}{RESET}"

            output.append(f"  {BOLD}Daemons:{RESET} hl_executor: {p_hl} | perp_ls: {p_perp} | pairs_arb: {p_pairs} | watchdog: {p_wd}")
            output.append(f"  {BOLD}Laya AI:{RESET} {p_laya} | Decisions: {BOLD}{laya_decisions:,}{RESET} | Feedback: {BOLD}{laya_feedback}{RESET}")
            output.append(f"{BOLD}{CYAN}╠══════════════════════════════════════════════════════════════════════════════════════╣{RESET}")

            # Positions Table
            output.append(f"  {BOLD}ACTIVE POSITIONS ({len(positions)}):{RESET}")
            output.append(f"  {'COIN':<8} {'SIDE':<6} {'SIZE':<12} {'ENTRY PX':<12} {'UNREALIZED PNL':<18} {'STATUS':<15}")
            output.append(f"  {'-'*8} {'-'*6} {'-'*12} {'-'*12} {'-'*18} {'-'*15}")

            for pos in positions:
                p = pos["position"]
                coin = p["coin"]
                szi = float(p["szi"])
                side = "LONG" if szi > 0 else "SHORT"
                entry_px = float(p.get("entryPx", 0))
                upnl = float(p.get("unrealizedPnl", 0))
                side_color = GREEN if side == "LONG" else MAGENTA
                upnl_str = format_currency(upnl)
                status = "In Profit 🟢" if upnl > 0 else "Stop Guarded 🛡️"
                output.append(f"  {BOLD}{coin:<8}{RESET} {side_color}{side:<6}{RESET} {abs(szi):<12.2f} ${entry_px:<11.4f} {upnl_str:<27} {status:<15}")

            output.append(f"{BOLD}{CYAN}╚══════════════════════════════════════════════════════════════════════════════════════╝{RESET}")
            output.append(f"  {YELLOW}Press Ctrl+C to exit dashboard | Refreshes every {interval}s{RESET}")

            print("\n".join(output), flush=True)

        except KeyboardInterrupt:
            print("\nExiting live console dashboard.")
            break
        except Exception as e:
            print(f"\n[Error updating dashboard: {e}]", flush=True)

        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=3, help="Refresh interval in seconds")
    args = parser.parse_args()
    run_dashboard(args.interval)
