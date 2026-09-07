"""
Backtest + Kelly-fraction sizing overlay for the cross-sectional perp long/short book
(src/execution/hl_perp_ls.py). Standalone research script -- does NOT import or modify
hl_perp_ls.py. Read-only vs live trading; writes a report only.

Replays the exact live strategy (72h momentum ranking, top/bottom 30% quantile,
4h rebalance, existing vol/DD scale overlay) over real Binance 1h history, then asks:
does adding a walk-forward Kelly-fraction multiplier on top of the existing vol/DD scale
improve risk-adjusted return vs the vol/DD-only baseline that's already live?

Kelly fraction here is estimated from the strategy's OWN trailing per-cycle return series
(not a coin-flip model): p = win rate of past cycles, b = avg_win/avg_loss payoff ratio,
kelly = p - (1-p)/b, clamped to [0, 1], then HALF-KELLY applied (financial-Kelly convention --
full Kelly is provably too aggressive under estimation error/fat tails).

Run: venv/bin/python -m src.quant.perp_ls_backtester
"""
import json
import os
import numpy as np
import pandas as pd
import ccxt
from src.utils.logger import log

UNIVERSE = ["BNB", "DOGE", "AVAX", "ARB", "OP", "NEAR", "ADA", "APT", "ATOM", "INJ"]
LOOKBACK_H = 72
QUANTILE = 0.30
HYSTERESIS = 0.15
REBALANCE_H = 4
TIMEFRAME = "1h"
CANDLES = 1000
FEE = 0.0005            # taker per side, matches backtester.py convention
TARGET_VOL = 0.008      # hourly, matches hl_perp_ls
DD_START, DD_MAX, EXPO_FLOOR = 0.05, 0.15, 0.20
KELLY_WINDOW = 20        # cycles of trailing history to estimate p/b (20*4h = ~80h)
KELLY_HALF = 0.5         # half-Kelly, standard practical discount


def fetch_closes(exchange):
    data = {}
    for base in UNIVERSE:
        sym = f"{base}/USDT"
        try:
            candles = exchange.fetch_ohlcv(sym, TIMEFRAME, limit=CANDLES)
            data[base] = pd.Series([c[4] for c in candles])
            log.info(f"[PERP_LS_BT] fetched {len(data[base])} {TIMEFRAME} candles for {sym}")
        except Exception as e:
            log.error(f"[PERP_LS_BT] fetch failed {sym}: {e}")
    return data


def rank_and_pick(mom_row):
    """mom_row: {base: momentum_return} for one cycle. Returns (longs, shorts)."""
    ranked = sorted(mom_row.items(), key=lambda kv: kv[1], reverse=True)
    n = max(1, int(len(ranked) * QUANTILE))
    longs = [b for b, _ in ranked[:n]]
    shorts = [b for b, _ in ranked[-n:]]
    return longs, shorts


def simulate(data):
    """Replays cross-sectional momentum L/S at 4h cadence. Returns per-cycle return series
    with and without the existing live vol/DD scale overlay (pre-Kelly baseline)."""
    df = pd.DataFrame(data).dropna()
    n_hours = len(df)
    cycle_starts = list(range(LOOKBACK_H, n_hours - REBALANCE_H, REBALANCE_H))

    raw_cycle_returns = []       # unscaled cross-sectional L/S return per cycle
    prev_book = None

    for t in cycle_starts:
        window = df.iloc[t - LOOKBACK_H:t]
        mom = {b: (window[b].iloc[-1] / window[b].iloc[0] - 1.0) for b in df.columns}
        longs, shorts = rank_and_pick(mom)

        entry = df.iloc[t]
        exitp = df.iloc[t + REBALANCE_H]
        gross_names = len(longs) + len(shorts)
        leg_w = 1.0 / gross_names if gross_names else 0.0

        ret = 0.0
        for b in longs:
            ret += leg_w * (exitp[b] / entry[b] - 1.0)
        for b in shorts:
            ret += leg_w * (1.0 - exitp[b] / entry[b])

        book = set(longs) | set(shorts)
        turnover = len(book ^ prev_book) / gross_names if prev_book else 1.0
        prev_book = book
        ret -= turnover * 2 * FEE   # round-trip fee on rotated names only (hysteresis-like)

        raw_cycle_returns.append(ret)

    return np.array(raw_cycle_returns)


def vol_dd_scale(cycle_returns):
    """Mirror hl_perp_ls's live composite_scale (vol target + smooth DD floor), walk-forward."""
    equity = [1.0]
    scaled_returns = []
    peak = 1.0
    trailing = []
    for r in cycle_returns:
        trailing.append(r)
        if len(trailing) > 6:
            trailing.pop(0)
        vol = float(np.std(trailing)) if len(trailing) > 1 else TARGET_VOL
        m_vol = 1.0 if vol <= 0 else min(1.0, TARGET_VOL / vol)

        cur_eq = equity[-1]
        peak = max(peak, cur_eq)
        max_dd = (peak - cur_eq) / peak if peak > 0 else 0.0
        if max_dd <= DD_START:
            m_dd = 1.0
        elif max_dd >= DD_MAX:
            m_dd = EXPO_FLOOR
        else:
            m_dd = 1.0 - ((max_dd - DD_START) / (DD_MAX - DD_START)) * (1.0 - EXPO_FLOOR)

        m = m_vol * m_dd
        scaled = r * m
        scaled_returns.append(scaled)
        equity.append(cur_eq * (1 + scaled))
    return np.array(scaled_returns), np.array(equity)


def kelly_overlay(cycle_returns):
    """Applies vol/DD scale AND a walk-forward half-Kelly multiplier estimated from the
    strategy's own trailing win rate / payoff ratio. Kelly reduces size when the recent
    edge looks weak/volatile, increases (up to 1x) when it's been reliably positive --
    this is a sizing overlay, not a new alpha source."""
    equity = [1.0]
    scaled_returns = []
    peak = 1.0
    trailing = []
    kelly_hist = []
    for r in cycle_returns:
        trailing.append(r)
        if len(trailing) > 6:
            trailing.pop(0)
        vol = float(np.std(trailing)) if len(trailing) > 1 else TARGET_VOL
        m_vol = 1.0 if vol <= 0 else min(1.0, TARGET_VOL / vol)

        cur_eq = equity[-1]
        peak = max(peak, cur_eq)
        max_dd = (peak - cur_eq) / peak if peak > 0 else 0.0
        if max_dd <= DD_START:
            m_dd = 1.0
        elif max_dd >= DD_MAX:
            m_dd = EXPO_FLOOR
        else:
            m_dd = 1.0 - ((max_dd - DD_START) / (DD_MAX - DD_START)) * (1.0 - EXPO_FLOOR)

        # Kelly fraction from trailing KELLY_WINDOW cycles (needs enough history; else 1.0 = no-op)
        if len(kelly_hist) >= KELLY_WINDOW:
            hist = np.array(kelly_hist[-KELLY_WINDOW:])
            wins = hist[hist > 0]
            losses = hist[hist < 0]
            p = len(wins) / len(hist) if len(hist) else 0.5
            avg_win = wins.mean() if len(wins) else 0.0
            avg_loss = abs(losses.mean()) if len(losses) else 1e-9
            b = avg_win / avg_loss if avg_loss > 0 else 1.0
            kelly_raw = p - (1 - p) / b if b > 0 else 0.0
            kelly = float(np.clip(kelly_raw, 0.0, 1.0)) * KELLY_HALF * 2  # half-kelly, capped [0,1]
            kelly = float(np.clip(kelly, 0.0, 1.0))
        else:
            kelly = 1.0   # not enough history yet -- defer to vol/DD scale only

        m = m_vol * m_dd * kelly
        scaled = r * m
        scaled_returns.append(scaled)
        equity.append(cur_eq * (1 + scaled))
        kelly_hist.append(r)
    return np.array(scaled_returns), np.array(equity)


def metrics(returns, equity):
    if len(returns) == 0:
        return {"cycles": 0}
    curve = np.array(equity)
    peak = np.maximum.accumulate(curve)
    dd = (curve - peak) / peak
    sharpe = float(returns.mean() / returns.std() * np.sqrt(len(returns))) if returns.std() > 0 else 0.0
    downside = returns[returns < 0]
    dd_dev = np.sqrt(np.mean(downside ** 2)) if len(downside) else 1e-9
    sortino = float(returns.mean() / dd_dev * np.sqrt(len(returns))) if dd_dev > 0 else 0.0
    return {
        "cycles": len(returns),
        "total_return_pct": round(float((curve[-1] - 1) * 100), 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "win_rate_pct": round(float((returns > 0).mean() * 100), 1),
        "max_dd_pct": round(float(dd.min() * 100), 2),
        "avg_cycle_return_pct": round(float(returns.mean() * 100), 4),
    }


def run():
    exchange = ccxt.binance()
    data = fetch_closes(exchange)
    if len(data) < 6:
        log.error("[PERP_LS_BT] insufficient symbol data, aborting.")
        return

    raw = simulate(data)
    baseline_returns, baseline_eq = vol_dd_scale(raw)
    kelly_returns, kelly_eq = kelly_overlay(raw)

    base_m = metrics(baseline_returns, baseline_eq)
    kelly_m = metrics(kelly_returns, kelly_eq)

    report = {
        "universe": UNIVERSE,
        "candles_per_symbol": CANDLES,
        "baseline_vol_dd_scale_only": base_m,
        "with_kelly_overlay": kelly_m,
        "kelly_improves_sharpe": kelly_m["sharpe"] > base_m["sharpe"],
        "kelly_improves_dd": kelly_m["max_dd_pct"] > base_m["max_dd_pct"],  # less negative = better
        "verdict": None,
    }
    if kelly_m["sharpe"] > base_m["sharpe"] * 1.05 and kelly_m["max_dd_pct"] >= base_m["max_dd_pct"]:
        report["verdict"] = "PROMOTE: Kelly overlay improves Sharpe without worsening drawdown."
    elif kelly_m["sharpe"] > base_m["sharpe"]:
        report["verdict"] = "MARGINAL: Kelly improves Sharpe but check drawdown tradeoff before promoting."
    else:
        report["verdict"] = "NO EDGE: Kelly overlay does not beat existing vol/DD scale. Do not promote."

    os.makedirs("models_local", exist_ok=True)
    with open("models_local/perp_ls_kelly_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n===== PERP L/S KELLY OVERLAY BACKTEST =====")
    print(f"cycles={base_m['cycles']} (4h each, ~{base_m['cycles']*4/24:.0f} days)")
    print(f"BASELINE (live vol/DD scale only): sharpe={base_m['sharpe']} sortino={base_m['sortino']} "
          f"return={base_m['total_return_pct']}% maxDD={base_m['max_dd_pct']}% win={base_m['win_rate_pct']}%")
    print(f"WITH KELLY OVERLAY:                 sharpe={kelly_m['sharpe']} sortino={kelly_m['sortino']} "
          f"return={kelly_m['total_return_pct']}% maxDD={kelly_m['max_dd_pct']}% win={kelly_m['win_rate_pct']}%")
    print(f"\nVERDICT: {report['verdict']}")
    print(f"\nWrote models_local/perp_ls_kelly_report.json")
    return report


if __name__ == "__main__":
    run()
