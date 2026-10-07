#!/usr/bin/env python3
"""Continuous terminal monitoring report for TradingBingx.

Usage: python3 scripts/continuous_monitor.py [--interval SECONDS] [--frames N]
Prints a timestamped health + P&L report every interval. No screen clearing,
so it scrolls cleanly in a terminal or screen/tmux session.
"""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VENV_PYTHON = os.path.join(_PROJECT_ROOT, "venv", "bin", "python3")
if os.path.exists(_VENV_PYTHON) and sys.prefix != os.path.join(_PROJECT_ROOT, "venv"):
    os.execv(_VENV_PYTHON, [_VENV_PYTHON] + sys.argv)

import argparse
import glob
import subprocess
import time
from datetime import datetime

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

import redis
import requests


def _silence_lib_logs():
    """Importing src.* re-runs setup_logger(); strip handlers after every import."""
    try:
        from loguru import logger as _grl
        _grl.remove()
    except Exception:
        pass

LOG = os.path.join(_PROJECT_ROOT, "logs", "polyarb.log")
PERP_LOG = os.path.join(_PROJECT_ROOT, "logs", "perp_ls.log")
WATCHDOG_LOG = os.path.join(_PROJECT_ROOT, "logs", "watchdog.log")


def main_log() -> str:
    """Live supervisor log; loguru rotates polyarb.log into timestamped files."""
    try:
        cands = glob.glob(os.path.join(_PROJECT_ROOT, "logs", "polyarb*.log"))
        if cands:
            return max(cands, key=os.path.getmtime)
    except Exception:
        pass
    return LOG

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def proc_running(name: str) -> bool:
    try:
        return subprocess.run(["pgrep", "-f", name], capture_output=True).returncode == 0
    except Exception:
        return False


def tail_new(path: str, marker: float, patterns) -> list:
    """Return matching log lines with timestamp > marker (dedup ms copies)."""
    out, last_ts = [], marker
    try:
        with open(path, "r", errors="replace") as f:
            for line in f:
                if not line[:4].isdigit():
                    continue
                try:
                    ts = datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S").timestamp()
                except ValueError:
                    continue
                if ts <= marker:
                    continue
                if any(p in line for p in patterns):
                    stripped = line.rstrip()
                    # drop the duplicate handler copy (same second, same text, ms suffix variant)
                    if out and out[-1][1][:19] == stripped[:19] and out[-1][1][20:].split(" - ")[-1] == stripped[20:].split(" - ")[-1]:
                        continue
                    out.append((ts, stripped))
                if ts > last_ts:
                    last_ts = ts
    except FileNotFoundError:
        pass
    return out, last_ts


def report(r, since: float) -> tuple:
    lines = []
    now = datetime.now()

    # --- capital ---
    try:
        cur = float(r.get("account_current_value") or 0)
        peak = float(r.get("account_peak_value") or 0)
        dd = ((peak - cur) / peak * 100) if peak > 0 else 0.0
        day_key = f"session_realized_pnl:{int(time.time() // 86400)}"
        pnl = float(r.get(day_key) or 0)
        open_n = int(r.get("open_positions_count") or 0)
    except Exception as e:
        lines.append(f"{RED}REDIS ERROR: {e}{RESET}")
        return lines, since

    eq_c = GREEN if cur >= peak * 0.92 else (YELLOW if cur >= peak * 0.85 else RED)
    pnl_c = GREEN if pnl >= 0 else RED
    dd_c = GREEN if dd < 10 else (YELLOW if dd < 15 else RED)

    lines.append(f"{BOLD}{CYAN}===== TRADINGBINGX MONITOR | {now.strftime('%Y-%m-%d %H:%M:%S')} ====={RESET}")
    lines.append(f"  Equity:   {eq_c}${cur:.2f}{RESET}   Peak: ${peak:.2f}   DD: {dd_c}{dd:.1f}%{RESET}")
    lines.append(f"  Session:  {pnl_c}${pnl:+.2f}{RESET}   Open positions: {open_n}")

    # --- daemons ---
    daemons = [
        ("hl_executor", "hl_executor.py"),
        ("streamer", "multi_streamer"),
        ("perp_ls", "hl_perp_ls"),
        ("laya", "laya_ai/.venv/bin/python3 server.py"),
    ]
    parts = []
    for label, pat in daemons:
        ok = proc_running(pat)
        parts.append(f"{label}: {GREEN}UP{RESET}" if ok else f"{label}: {RED}DOWN{RESET}")
    laya_detail = ""
    try:
        h = requests.get("http://127.0.0.1:8080/health", timeout=2).json()
        laya_detail = f" (v{h.get('laya_version', '?')}, decisions={h.get('total_historical_decisions', 0):,})"
        parts[-1] = f"laya: {GREEN}UP{RESET}"
    except Exception:
        parts[-1] = f"laya: {RED}HTTP DOWN{RESET}"
    lines.append(f"  Daemons:  {' | '.join(parts)}{laya_detail}")

    try:
        wd_ok = subprocess.run(["crontab", "-l"], capture_output=True, text=True).stdout.count("TradingBingx") >= 1
    except Exception:
        wd_ok = False
    halt = os.path.exists(os.path.join(_PROJECT_ROOT, "state", "STALE_WAKE_HALT"))
    lines.append(f"  Cron:     {'watchdog OK' if wd_ok else 'watchdog MISSING'}"
                 f" | halt file: {'PRESENT' if halt else 'clear'}")

    # --- alerts ---
    alerts = []
    if dd >= 15:
        alerts.append(f"{RED}CRITICAL: drawdown {dd:.1f}% >= 15%{RESET}")
    elif dd >= 10:
        alerts.append(f"{YELLOW}WARNING: drawdown {dd:.1f}% >= 10%{RESET}")
    if pnl <= -5:
        alerts.append(f"{YELLOW}SESSION_GUARD: ${pnl:+.2f} <= -$5 -> new entries BLOCKED{RESET}")
    if open_n >= 4:
        alerts.append(f"{YELLOW}position cap {open_n}/4{RESET}")
    for label, pat in daemons:
        if not proc_running(pat):
            alerts.append(f"{RED}{label} DOWN{RESET}")
    if halt:
        alerts.append(f"{RED}STALE_WAKE_HALT present{RESET}")
    lines.append(f"  Alerts:   {'; '.join(alerts) if alerts else GREEN + 'none' + RESET}")

    # --- positions (from DB) ---
    try:
        import contextlib
        import io
        from src.utils.db import DatabaseManager
        _silence_lib_logs()  # import re-added loguru stdout handler; drop it again
        _buf = io.StringIO()
        with contextlib.redirect_stdout(_buf), contextlib.redirect_stderr(_buf):  # suppress db connect noise
            db = DatabaseManager()
            pos = {k: v for k, v in (db.get_positions() or {}).items() if v.get("quantity")}
        if pos:
            lines.append(f"  Positions:")
            for (sym, ex), v in sorted(pos.items()):
                lines.append(f"    {sym:<12} qty={v.get('quantity')} avg={v.get('avg_price')}")
        else:
            lines.append(f"  Positions: flat")
    except Exception as e:
        lines.append(f"  Positions: {YELLOW}db err {e}{RESET}")

    # --- new log activity since last frame ---
    pats = ["Decision:", "LIVE ORDER", "SESSION_GUARD", "ERROR", "Laya veto",
            "LAYA_VETO", "DD KILL", "closed", "CLOSED", "SETTLEMENT]", "EDGE_GATE"]
    new_main, ts1 = tail_new(main_log(), since, pats)
    new_perp, ts2 = tail_new(PERP_LOG, since, ["DD KILL", "halted by kill-switch",
                                               "resuming trading", "flatten", "ERROR"])
    since = max(since, ts1, ts2)
    evts = [l for _, l in (new_main + new_perp)[-12:]]
    # dedup the double-handler lines (ms-suffixed copies)
    dedup = []
    for l in evts:
        if dedup and dedup[-1][:19] == l[:19] and dedup[-1][20:].split(" - ")[-1] == l[20:].split(" - ")[-1]:
            continue
        dedup.append(l)
    if dedup:
        lines.append(f"  Activity ({len(dedup)} new):")
        for l in dedup:
            if "ERROR" in l or "DD KILL" in l:
                c = RED
            elif "SESSION_GUARD" in l or "WARNING" in l:
                c = YELLOW
            elif "LIVE ORDER" in l:
                c = GREEN
            else:
                c = RESET
            # shorten: drop module path noise
            body = l.split(" - ", 1)[-1] if " - " in l else l
            lines.append(f"    {c}{l[11:19]} {body}{RESET}")
    else:
        lines.append(f"  Activity: no new events this frame")

    return lines, since


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=15)
    ap.add_argument("--frames", type=int, default=0, help="0 = run forever")
    args = ap.parse_args()

    try:
        r = redis.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, decode_responses=True)
        r.ping()
    except Exception as e:
        print(f"{RED}Cannot connect to Redis: {e}{RESET}")
        return 1

    since = time.time() - 5
    frames = 0
    while True:
        try:
            lines, since = report(r, since)
            print("\n".join(lines), flush=True)
        except KeyboardInterrupt:
            print("\nMonitor stopped.")
            return 0
        except Exception as e:
            print(f"{YELLOW}[frame error: {e}]{RESET}", flush=True)
        frames += 1
        if args.frames and frames >= args.frames:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main() or 0)
