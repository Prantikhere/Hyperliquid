"""
Cross-sectional long-short research + walk-forward OOS validation.

The price-technical stack (backtester.py) trades each symbol in isolation and showed no
out-of-sample edge. This module tests a different, better-documented family: CROSS-SECTIONAL
strategies that rank the whole universe each rebalance and go long the top / short the bottom,
dollar-neutral. Market beta ~= 0, so the return is the *spread* between winners and losers, not
market direction -- the "steady, market-neutral" property we actually want.

Two signal families, both classic and both real anomalies in crypto:
  - MOMENTUM : long recent out-performers, short under-performers (medium horizon).
  - REVERSAL : long recent losers, short winners (short horizon; strong in crypto).

Honesty rules (identical philosophy to walkforward.py):
  - Params (lookback, hold, quantile, mode) are chosen ONLY on the train segment.
  - The chosen config is then measured on the immediately-following test segment it never saw.
  - OOS returns are concatenated across folds. We report OOS Sharpe, annualized return, max
    drawdown and monthly mean -- the numbers that decide if this is real.
  - Realistic cost: taker fee per leg * turnover each rebalance.

Run: venv/bin/python -m src.quant.xs_research
Writes: models_local/xs_research_report.json
"""
import json
import os
import time
import numpy as np
import ccxt
from src.quant.backtester import SYMBOLS
from src.utils.logger import log

TIMEFRAME = "1d"
HISTORY_DAYS = 1400            # ~3.8 years
FEE = 0.0005                   # 5 bps taker per side
# Walk-forward geometry (in daily bars).
TRAIN_LEN = 365
TEST_LEN = 120
FIRST_TEST_START = 365

# Parameter grid searched on TRAIN each fold.
#   mode: "MOM" (long winners) or "REV" (long losers)
#   lookback: signal formation window (days)
#   hold: rebalance interval (days) -- position held this long, then re-ranked
#   quantile: fraction of universe on each side (0.2 = long top 20%, short bottom 20%)
GRID = []
for mode in ("MOM", "REV"):
    for lookback in ((30, 60, 90) if mode == "MOM" else (2, 3, 5)):
        for hold in ((7, 14) if mode == "MOM" else (1, 2, 3)):
            for q in (0.2, 0.3):
                GRID.append({"mode": mode, "lookback": lookback, "hold": hold, "quantile": q})

ANN_DAYS = 365                 # crypto trades every day


def fetch_daily(symbol, exchange, days=HISTORY_DAYS):
    """Paginate daily closes back `days` bars. Returns (timestamps_ms, closes)."""
    limit = 1000
    since = exchange.milliseconds() - days * 24 * 3600 * 1000
    rows = []
    while True:
        batch = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=since, limit=limit)
        if not batch:
            break
        rows += batch
        since = batch[-1][0] + 24 * 3600 * 1000
        if len(batch) < limit or len(rows) >= days:
            break
        time.sleep(exchange.rateLimit / 1000)
    # dedupe by timestamp, keep chronological
    seen = {}
    for r in rows:
        seen[r[0]] = r[4]
    ts = sorted(seen)
    return ts, [seen[t] for t in ts]


def build_matrix(data):
    """Align symbols on the common timestamp grid. Returns (symbols, ts, price[T,N])."""
    # UNION of timestamps, not intersection: newer-listed coins simply carry NaN before their
    # listing and are excluded from those cross-sections (simulate_xs NaN-masks per day). Using
    # intersection would collapse the series to the shortest-history symbol.
    common = set()
    for ts, _ in data.values():
        common |= set(ts)
    common = sorted(common)
    syms = list(data.keys())
    T, N = len(common), len(syms)
    idx = {t: i for i, t in enumerate(common)}
    px = np.full((T, N), np.nan)
    for j, s in enumerate(syms):
        ts, closes = data[s]
        for t, c in zip(ts, closes):
            if t in idx:
                px[idx[t], j] = c
    return syms, common, px


def simulate_xs(px, cfg, seg_start, seg_end):
    """
    Dollar-neutral cross-sectional long-short over [seg_start, seg_end).
    Feeds `lookback` bars of prior history so the first rebalance has a full signal window.
    Returns list of per-rebalance net returns (fraction of gross capital).
    """
    mode, L, H, q = cfg["mode"], cfg["lookback"], cfg["hold"], cfg["quantile"]
    T, N = px.shape
    rets = []
    prev_w = np.zeros(N)
    t = seg_start
    while t + H < seg_end:
        if t - L < 0:
            t += H
            continue
        past = px[t - L, :]
        now = px[t, :]
        fut = px[t + H, :]
        valid = np.isfinite(past) & np.isfinite(now) & np.isfinite(fut) & (past > 0) & (now > 0)
        if valid.sum() < 6:                     # need a real cross-section
            t += H
            continue
        sig = np.full(N, np.nan)
        sig[valid] = now[valid] / past[valid] - 1.0     # formation return
        if mode == "REV":
            sig = -sig                          # long losers
        order = np.argsort(np.where(np.isfinite(sig), sig, -np.inf))
        v = order[np.isfinite(sig[order])]
        k = max(1, int(len(v) * q))
        longs = v[-k:]
        shorts = v[:k]
        w = np.zeros(N)
        w[longs] = 0.5 / k                       # gross 1.0: 0.5 long + 0.5 short
        w[shorts] = -0.5 / k
        # forward return of the held book
        fwd = fut[valid] / now[valid] - 1.0
        r = float(np.nansum(w[valid] * fwd))
        # turnover cost vs previous book
        turnover = float(np.abs(w - prev_w).sum())
        r -= turnover * FEE
        rets.append(r)
        prev_w = w
        t += H
    return rets


def perf(rets, hold):
    """Annualized metrics from a list of per-rebalance returns spaced `hold` days apart."""
    if not rets:
        return {"n": 0, "sharpe": 0.0, "ann_return_pct": 0.0, "monthly_pct": 0.0, "max_dd_pct": 0.0, "win_rate": 0.0}
    a = np.array(rets)
    eq = np.cumprod(1 + a)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    per_year = ANN_DAYS / hold
    mean, sd = a.mean(), a.std()
    sharpe = float(mean / sd * np.sqrt(per_year)) if sd > 0 else 0.0
    ann = float(eq[-1] ** (per_year / len(a)) - 1) if len(a) > 0 and eq[-1] > 0 else -1.0
    return {
        "n": len(a),
        "sharpe": round(sharpe, 3),
        "ann_return_pct": round(ann * 100, 1),
        "monthly_pct": round(((1 + ann) ** (1 / 12) - 1) * 100, 2) if ann > -1 else -100.0,
        "max_dd_pct": round(float(dd.min()) * 100, 1),
        "win_rate": round(float((a > 0).mean()) * 100, 1),
    }


def make_folds(T):
    folds = []
    ts = FIRST_TEST_START
    while ts + TEST_LEN <= T and ts - TRAIN_LEN >= 0:
        folds.append({"train": (ts - TRAIN_LEN, ts), "test": (ts, ts + TEST_LEN)})
        ts += TEST_LEN
    return folds


def pick_cfg_on_train(px, tr0, tr1):
    """Best config by train Sharpe."""
    best, best_s = GRID[0], -1e9
    for cfg in GRID:
        m = perf(simulate_xs(px, cfg, tr0, tr1), cfg["hold"])
        if m["sharpe"] > best_s:
            best_s, best = m["sharpe"], cfg
    return best, round(best_s, 3)


def run():
    exchange = ccxt.binance()
    data = {}
    for s in SYMBOLS:
        try:
            ts, closes = fetch_daily(s, exchange)
            if len(ts) > 200:
                data[s] = (ts, closes)
                log.info(f"XS fetched {len(ts)} daily bars {s}")
        except Exception as e:
            log.error(f"XS fetch failed {s}: {e}")
    if len(data) < 8:
        print(f"Only {len(data)} symbols fetched; need >=8 for a cross-section. Abort.")
        return

    syms, ts, px = build_matrix(data)
    T = px.shape[0]
    folds = make_folds(T)
    log.info(f"XS: {len(syms)} symbols, {T} common days, {len(folds)} folds")
    if not folds:
        print("Series too short for walk-forward. Abort.")
        return

    oos_rets_by_hold = {}          # need consistent hold for annualization; store (r, hold)
    oos_rets = []
    oos_holds = []
    fold_log = []
    for fi, fold in enumerate(folds):
        tr0, tr1 = fold["train"]
        te0, te1 = fold["test"]
        cfg, train_s = pick_cfg_on_train(px, tr0, tr1)
        te_rets = simulate_xs(px, cfg, te0, te1)
        oos_rets.extend(te_rets)
        oos_holds.extend([cfg["hold"]] * len(te_rets))
        m = perf(te_rets, cfg["hold"])
        fold_log.append({
            "fold": fi, "train": [tr0, tr1], "test": [te0, te1],
            "chosen": cfg, "train_sharpe": train_s,
            "test_sharpe": m["sharpe"], "test_ann_pct": m["ann_return_pct"], "test_n": m["n"],
        })
        log.info(f"fold {fi}: pick {cfg} train_sharpe={train_s} -> test_sharpe={m['sharpe']} ann={m['ann_return_pct']}%")

    # Aggregate OOS. Use median hold for annualization scale (folds may pick different holds).
    med_hold = int(np.median(oos_holds)) if oos_holds else 7
    agg = perf(oos_rets, med_hold)

    report = {
        "method": "cross-sectional dollar-neutral long-short, walk-forward OOS",
        "universe_size": len(syms),
        "common_days": T,
        "n_folds": len(folds),
        "oos": agg,
        "median_hold_days": med_hold,
        "folds": fold_log,
    }
    os.makedirs("models_local", exist_ok=True)
    with open("models_local/xs_research_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n===== CROSS-SECTIONAL LONG-SHORT — WALK-FORWARD OOS =====")
    print(f"universe={len(syms)} days={T} folds={len(folds)}")
    for fl in fold_log:
        c = fl["chosen"]
        print(f"  fold {fl['fold']}: {c['mode']} L={c['lookback']} H={c['hold']} q={c['quantile']} "
              f"| train_sh={fl['train_sharpe']:+.2f} -> TEST sh={fl['test_sharpe']:+.2f} "
              f"ann={fl['test_ann_pct']:+.1f}% n={fl['test_n']}")
    print("\n  ---- AGGREGATE OUT-OF-SAMPLE ----")
    print(f"  Sharpe        : {agg['sharpe']}")
    print(f"  Ann return    : {agg['ann_return_pct']}%")
    print(f"  Monthly (comp): {agg['monthly_pct']}%")
    print(f"  Max drawdown  : {agg['max_dd_pct']}%")
    print(f"  Win rate      : {agg['win_rate']}%  over n={agg['n']} rebalances")
    verdict = "REAL EDGE (OOS)" if agg["sharpe"] > 0.8 and agg["ann_return_pct"] > 0 else "NO OOS EDGE"
    print(f"\n  VERDICT: {verdict}")
    print("\nWrote models_local/xs_research_report.json")
    return report


if __name__ == "__main__":
    run()
