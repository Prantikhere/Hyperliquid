"""
Cointegrated pairs stat-arb research + backtest. SHADOW/RESEARCH ONLY -- standalone script,
not imported by any live process, writes a report only. Different alpha source from the
existing momentum L/S book (hl_perp_ls) and mean-reversion/momentum ensemble (supervisor.py):
trades the SPREAD between two cointegrated assets, market-neutral on the pair itself.

Method:
  1. Fetch 1h closes for a broad symbol universe (Binance).
  2. Engle-Granger cointegration test (statsmodels) on every pair, keep p < COINT_PVALUE.
  3. For each surviving pair: OLS hedge ratio, z-score the spread, walk-forward backtest --
     enter when |z| > ENTRY_Z, exit when |z| < EXIT_Z or z flips sign (mean reversion done).
  4. Report per-pair and aggregate metrics. No promotion to live trading in this file.

Run: venv/bin/python -m src.quant.pairs_stat_arb
"""
import json
import os
import itertools
import numpy as np
import pandas as pd
import ccxt
from statsmodels.tsa.stattools import coint
from src.utils.logger import log

SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "DOGE/USDT", "AVAX/USDT",
    "ADA/USDT", "SUI/USDT", "APT/USDT", "ARB/USDT", "OP/USDT", "ATOM/USDT",
    "NEAR/USDT", "TIA/USDT", "INJ/USDT", "LDO/USDT", "AAVE/USDT", "DYDX/USDT",
    "ETC/USDT", "FIL/USDT",
]
TIMEFRAME = "1h"
CANDLES = 4000          # ~167 days; paginated (binance 1h hard cap is 1000/request)
COINT_PVALUE = 0.05
LOOKBACK = 200          # rolling window for hedge ratio + z-score re-estimation
ENTRY_Z = 2.0
EXIT_Z = 0.5
FEE = 0.0005            # taker per side, per leg
MAX_HOLD_H = 96         # force-exit stale spreads after 4 days


def fetch_ohlcv_paginated(exchange, sym, total, batch=1000):
    """Binance caps 1h candles at 1000/request. Walk backward in time via `since` to build
    a longer history, oldest-first."""
    all_candles = []
    end_ts = exchange.milliseconds()
    ms_per_candle = 3600 * 1000
    while len(all_candles) < total:
        remaining = total - len(all_candles)
        n = min(batch, remaining)
        since = end_ts - n * ms_per_candle
        chunk = exchange.fetch_ohlcv(sym, TIMEFRAME, since=since, limit=n)
        if not chunk:
            break
        all_candles = chunk + all_candles
        end_ts = chunk[0][0] - ms_per_candle
    # de-dupe by timestamp, sort ascending
    seen = {}
    for c in all_candles:
        seen[c[0]] = c
    ordered = [seen[k] for k in sorted(seen)]
    return ordered[-total:] if len(ordered) > total else ordered


def fetch_closes(exchange):
    data = {}
    for sym in SYMBOLS:
        try:
            candles = fetch_ohlcv_paginated(exchange, sym, CANDLES)
            data[sym] = pd.Series([c[4] for c in candles])
            log.info(f"[PAIRS] fetched {len(data[sym])} {TIMEFRAME} candles for {sym}")
        except Exception as e:
            log.error(f"[PAIRS] fetch failed {sym}: {e}")
    return data


def find_cointegrated_pairs(data):
    df = pd.DataFrame(data).dropna()
    pairs = []
    for a, b in itertools.combinations(df.columns, 2):
        try:
            _, pvalue, _ = coint(df[a], df[b])
            if pvalue < COINT_PVALUE:
                pairs.append((a, b, float(pvalue)))
        except Exception as e:
            log.debug(f"[PAIRS] coint test failed {a}/{b}: {e}")
    pairs.sort(key=lambda p: p[2])
    return pairs, df


def backtest_pair(df, a, b):
    """Walk-forward: re-fit hedge ratio + z-score stats every LOOKBACK bars, trade the spread."""
    n = len(df)
    trades = []
    pos = 0          # 0 flat, +1 long-spread (long a, short b), -1 short-spread
    entry_z = 0.0
    entry_idx = 0
    hedge = 1.0

    for i in range(LOOKBACK, n):
        window_a = df[a].iloc[i - LOOKBACK:i]
        window_b = df[b].iloc[i - LOOKBACK:i]
        # OLS hedge ratio: a = hedge*b + c
        hedge = float(np.polyfit(window_b, window_a, 1)[0])
        spread = window_a - hedge * window_b
        mu, sigma = spread.mean(), spread.std()
        if sigma == 0:
            continue

        cur_spread = df[a].iloc[i] - hedge * df[b].iloc[i]
        z = (cur_spread - mu) / sigma

        if pos == 0:
            if z > ENTRY_Z:
                pos, entry_z, entry_idx = -1, z, i   # spread too high -> short spread (short a, long b)
            elif z < -ENTRY_Z:
                pos, entry_z, entry_idx = 1, z, i    # spread too low -> long spread (long a, short b)
        else:
            held = i - entry_idx
            mean_reverted = (pos == 1 and z >= -EXIT_Z) or (pos == -1 and z <= EXIT_Z)
            stale = held >= MAX_HOLD_H
            flipped = (pos == 1 and z > ENTRY_Z) or (pos == -1 and z < -ENTRY_Z)
            if mean_reverted or stale or flipped:
                ret_a = df[a].iloc[i] / df[a].iloc[entry_idx] - 1.0
                ret_b = df[b].iloc[i] / df[b].iloc[entry_idx] - 1.0
                # pos=1: long a, short b (equal-risk weighted by hedge ratio)
                pnl = pos * (ret_a - hedge * ret_b) / (1 + abs(hedge)) - 2 * FEE
                trades.append(pnl)
                pos = 0

    return trades


def metrics(trades):
    if not trades:
        return {"trades": 0, "sharpe": 0.0, "win_rate_pct": 0.0, "total_return_pct": 0.0, "max_dd_pct": 0.0}
    arr = np.array(trades)
    eq = np.cumprod(1 + arr)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    sharpe = float(arr.mean() / arr.std() * np.sqrt(len(arr))) if arr.std() > 0 else 0.0
    return {
        "trades": len(trades),
        "sharpe": round(sharpe, 3),
        "win_rate_pct": round(float((arr > 0).mean() * 100), 1),
        "total_return_pct": round(float((eq[-1] - 1) * 100), 2),
        "max_dd_pct": round(float(dd.min() * 100), 2),
    }


def run():
    exchange = ccxt.binance()
    data = fetch_closes(exchange)
    if len(data) < 4:
        log.error("[PAIRS] insufficient symbol data, aborting.")
        return

    pairs, df = find_cointegrated_pairs(data)
    log.info(f"[PAIRS] {len(pairs)}/{len(list(itertools.combinations(data.keys(), 2)))} pairs cointegrated (p<{COINT_PVALUE})")

    results = []
    for a, b, pval in pairs:
        trades = backtest_pair(df, a, b)
        m = metrics(trades)
        m.update({"pair": f"{a}/{b}", "coint_pvalue": round(pval, 4)})
        results.append(m)

    results.sort(key=lambda r: r["sharpe"], reverse=True)

    MIN_TRADES = 15  # widened window -> require more trades to reduce small-n overfitting risk
    validated = [r for r in results if r["trades"] >= MIN_TRADES and r["sharpe"] > 0.5]
    verdict = (
        f"PROMOTE-CANDIDATE: {len(validated)} pair(s) show sharpe>0.5 on >={MIN_TRADES} trades."
        if validated else
        "NO EDGE: no pair clears sharpe>0.5 with enough trades. Do not promote."
    )

    report = {
        "universe": SYMBOLS,
        "candles_per_symbol": CANDLES,
        "cointegrated_pairs_found": len(pairs),
        "results": results,
        "validated_pairs": validated,
        "verdict": verdict,
    }
    os.makedirs("models_local", exist_ok=True)
    with open("models_local/pairs_stat_arb_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n===== PAIRS STAT-ARB BACKTEST =====")
    print(f"cointegrated pairs (p<{COINT_PVALUE}): {len(pairs)}")
    for r in results[:15]:
        print(f"  {r['pair']:20s} coint_p={r['coint_pvalue']:.4f} sharpe={r['sharpe']:+.2f} "
              f"win={r['win_rate_pct']}% ret={r['total_return_pct']:+.1f}% dd={r['max_dd_pct']:.1f}% n={r['trades']}")
    print(f"\nVERDICT: {verdict}")
    print(f"\nWrote models_local/pairs_stat_arb_report.json")
    return report


if __name__ == "__main__":
    run()
