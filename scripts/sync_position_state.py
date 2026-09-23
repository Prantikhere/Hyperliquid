"""Rebuild Redis position counters from on-chain + DB truth.

Fixes stale keys left behind by failed exits / crashes / manual closes:
  - open_positions_count
  - symbol_positions:* (phantom OP=1 etc.)
  - position_age:* for closed symbols
  - peak_roi:* for closed symbols
Run anytime; safe to re-run.
"""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

import redis as redis_lib
from src.execution.hl_raw import HlSdkClient
from src.utils.db import DatabaseManager


def main():
    r = redis_lib.Redis(host=os.getenv("REDIS_HOST", "localhost"), port=6379, decode_responses=True)
    db = DatabaseManager()
    client = HlSdkClient()

    # On-chain truth (primary source)
    onchain = client.get_positions()
    open_symbols = set()
    for p in onchain:
        pos = p.get("position", p)
        coin = pos.get("coin", "")
        szi = float(pos.get("szi", 0) or 0)
        if coin and szi != 0:
            open_symbols.add(f"{coin}/USDT")

    print(f"On-chain open symbols: {sorted(open_symbols) or '(none)'}")

    # DB truth (should match after next reconcile)
    db_pos = db.get_positions()
    db_open = {sym for (sym, eid), info in db_pos.items()
               if eid == "hyperliquid" and info.get("quantity", 0) != 0}
    print(f"DB open symbols: {sorted(db_open) or '(none)'}")

    # Prefer on-chain; if empty fall back to DB
    truth = open_symbols or db_open

    cleaned = []
    for key in list(r.scan_iter("symbol_positions:*")):
        # key format: symbol_positions:{exchange}:{SYMBOL/USDT}
        parts = key.split(":", 2)
        if len(parts) != 3:
            continue
        _, eid, symbol = parts
        if eid != "hyperliquid":
            continue
        if symbol in truth:
            r.set(key, 1)
            print(f"  KEEP  {key}=1")
        else:
            r.delete(key)
            cleaned.append(key)
            print(f"  DEL   {key}")

    for pattern in ("position_age:hyperliquid:*", "peak_roi:hyperliquid:*"):
        for key in list(r.scan_iter(pattern)):
            symbol = key.rsplit(":", 1)[-1]
            if symbol not in truth:
                r.delete(key)
                cleaned.append(key)
                print(f"  DEL   {key}")

    # Rebuild global count
    total = 0
    for key in r.scan_iter("symbol_positions:*"):
        try:
            total += int(r.get(key) or 0)
        except (TypeError, ValueError):
            pass
    r.set("open_positions_count", max(0, total))
    print(f"open_positions_count -> {max(0, total)}")
    print(f"Cleaned {len(cleaned)} stale keys")

    # Verify DB matches on-chain
    if open_symbols != db_open:
        print(f"NOTE: DB differs from on-chain — reconcile will fix on next cycle "
              f"(onchain={sorted(open_symbols)}, db={sorted(db_open)})")
    else:
        print("DB and on-chain agree")


if __name__ == "__main__":
    main()
