"""
Cross-sectional funding factor + CRASH/VOL DE-RISK OVERLAY — walk-forward OOS.

Base signal (xs_funding.py) earns positive carry but has a fat LEFT TAIL: 2 of 8 folds were
deleveraging events (-39%, -40% ann) that ate the whole edge. This module wraps the SAME base
factor in a risk overlay that cuts gross exposure exactly in those regimes, using only
information available AT each rebalance (no lookahead):

  1. Realized-vol targeting : exposure_vol = clip(target_vol / trailing_market_vol, 0, 1).
     When the market gets volatile, size down. Standard, non-exotic, hard to overfit.
  2. Drawdown cutout        : if the equal-weight market index is in a trailing drawdown deeper
     than dd_cut, go FLAT. Directly targets crash/deleveraging folds.

Final per-rebalance exposure m = exposure_vol, forced to 0 during a deep drawdown. Weights and
turnover are scaled by m; parked capital earns 0.

Honesty: base (quantile, hold) AND overlay (vol_lookback, target_vol, dd_cut) are ALL selected
on TRAIN by Sharpe, then measured on the unseen TEST fold. Reports straight OOS numbers.

Run: venv/bin/python -m src.quant.xs_funding_overlay
Writes: models_local/xs_funding_overlay_report.json
"""
import json
import os
import numpy as np
import ccxt
from src.quant.backtester import SYMBOLS
from src.quant.funding_carry import INTERVALS_PER_YEAR
from src.quant.xs_funding import (
    fetch_funding_ts, fetch_price_8h, build, perf, make_folds, FEE,
    TRAIN_LEN, TEST_LEN, FIRST_TEST_START,
)
from src.utils.logger import log

# Joint grid: base signal x overlay. Kept small to limit selection freedom (anti-overfit).
BASE = [{"quantile": 0.2, "hold": 3}, {"quantile": 0.3, "hold": 3}]
VOL_LOOKBACK = [12, 24]            # intervals (8h) for trailing realized vol
TARGET_VOL = [0.02, 0.04]         # per-8h vol target (~ maps to modest annualized)
DD_CUT = [None, 0.15, 0.25]       # flatten if market index drawdown deeper than this


def market_series(px):
    """Equal-weight market index return per interval + trailing drawdown, no lookahead.
    r_mkt[t] uses px[t]/px[t-1] (both known by t); dd[t] uses index up to t only."""
    T, N = px.shape
    r = np.zeros(T)
    for t in range(1, T):
        prev, now = px[t - 1, :], px[t, :]
        v = np.isfinite(prev) & np.isfinite(now) & (prev > 0)
        r[t] = float(np.mean(now[v] / prev[v] - 1.0)) if v.sum() else 0.0
    idx = np.cumprod(1 + r)
    peak = np.maximum.accumulate(idx)
    dd = idx / peak - 1.0
    return r, dd


def trailing_vol(r_mkt, L):
    """rv[t] = std of market returns over the trailing L intervals ending at t (known at t)."""
    T = len(r_mkt)
    rv = np.zeros(T)
    for t in range(T):
        lo = max(0, t - L + 1)
        seg = r_mkt[lo:t + 1]
        rv[t] = float(seg.std()) if len(seg) > 1 else 0.0
    return rv


def simulate_overlay(fund, px, cfg, s0, s1, rv, dd):
    """Base funding long-short scaled by the risk overlay. cfg carries base+overlay params."""
    q, H = cfg["quantile"], cfg["hold"]
    target, dd_cut = cfg["target_vol"], cfg["dd_cut"]
    T, N = fund.shape
    rets, exposures = [], []
    prev_w = np.zeros(N)
    t = s0
    while t + H < s1:
        fr = fund[t, :]
        now = px[t, :]
        fut = px[t + H, :]
        valid = np.isfinite(fr) & np.isfinite(now) & np.isfinite(fut) & (now > 0)
        if valid.sum() < 6:
            t += H
            continue
        v = np.where(valid)[0]
        order = v[np.argsort(fr[v])]
        k = max(1, int(len(order) * q))
        longs, shorts = order[:k], order[-k:]
        w = np.zeros(N)
        w[longs] = 0.5 / k
        w[shorts] = -0.5 / k

        # --- risk overlay (uses only rv[t], dd[t]; both trailing) ---
        m = 1.0 if rv[t] <= 0 else min(1.0, target / rv[t])
        if dd_cut is not None and dd[t] <= -dd_cut:
            m = 0.0
        w = w * m
        exposures.append(m)

        price_ret = np.zeros(N)
        price_ret[valid] = fut[valid] / now[valid] - 1.0
        fsum = np.zeros(N)
        for hh in range(H):
            row = fund[t + hh, :]
            fsum[valid] += np.where(np.isfinite(row[valid]), row[valid], 0.0)
        r = float(np.nansum(w[valid] * price_ret[valid]) - np.nansum(w[valid] * fsum[valid]))
        r -= float(np.abs(w - prev_w).sum()) * FEE
        rets.append(r)
        prev_w = w
        t += H
    return rets, exposures


def pick_cfg(fund, px, tr0, tr1, rv_cache, dd):
    best, best_s = None, -1e9
    for base in BASE:
        for L in VOL_LOOKBACK:
            rv = rv_cache[L]
            for tv in TARGET_VOL:
                for dc in DD_CUT:
                    cfg = {**base, "vol_lookback": L, "target_vol": tv, "dd_cut": dc}
                    rets, _ = simulate_overlay(fund, px, cfg, tr0, tr1, rv, dd)
                    m = perf(rets, base["hold"])
                    if m["sharpe"] > best_s:
                        best_s, best = m["sharpe"], cfg
    return best, round(best_s, 3)


def run():
    exchange = ccxt.binance({"options": {"defaultType": "future"}})
    data = {}
    for s in SYMBOLS:
        try:
            fmap = fetch_funding_ts(s, exchange)
            pmap = fetch_price_8h(s, exchange)
            if len(fmap) > 400 and len(pmap) > 400:
                data[s] = (fmap, pmap)
                log.info(f"OVL {s}: {len(fmap)} funding, {len(pmap)} price")
        except Exception as e:
            log.error(f"OVL fetch {s}: {e}")
    if len(data) < 8:
        print(f"Only {len(data)} symbols; need >=8. Abort.")
        return

    syms, buckets, fund, px = build(data)
    T = fund.shape[0]
    folds = make_folds(T)
    log.info(f"OVL: {len(syms)} symbols, {T} intervals, {len(folds)} folds")
    if not folds:
        print("Series too short. Abort.")
        return

    r_mkt, dd = market_series(px)
    rv_cache = {L: trailing_vol(r_mkt, L) for L in VOL_LOOKBACK}

    oos_rets, oos_holds, oos_expo, fold_log = [], [], [], []
    for fi, fold in enumerate(folds):
        tr0, tr1 = fold["train"]
        te0, te1 = fold["test"]
        cfg, train_s = pick_cfg(fund, px, tr0, tr1, rv_cache, dd)
        rv = rv_cache[cfg["vol_lookback"]]
        te, expo = simulate_overlay(fund, px, cfg, te0, te1, rv, dd)
        oos_rets.extend(te)
        oos_holds.extend([cfg["hold"]] * len(te))
        oos_expo.extend(expo)
        m = perf(te, cfg["hold"])
        fold_log.append({"fold": fi, "chosen": cfg, "train_sharpe": train_s,
                         "test_sharpe": m["sharpe"], "test_ann_pct": m["ann_return_pct"],
                         "test_n": m["n"], "avg_exposure": round(float(np.mean(expo)), 2) if expo else 0.0})
        log.info(f"fold {fi}: {cfg} train_sh={train_s} -> test_sh={m['sharpe']} ann={m['ann_return_pct']}% expo={np.mean(expo) if expo else 0:.2f}")

    med_hold = int(np.median(oos_holds)) if oos_holds else 3
    agg = perf(oos_rets, med_hold)
    agg["avg_exposure"] = round(float(np.mean(oos_expo)), 2) if oos_expo else 0.0

    report = {"method": "cross-sectional funding factor + vol-target/drawdown overlay, walk-forward OOS",
              "universe_size": len(syms), "intervals": T, "n_folds": len(folds),
              "oos": agg, "median_hold_intervals": med_hold, "folds": fold_log}
    os.makedirs("models_local", exist_ok=True)
    with open("models_local/xs_funding_overlay_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n===== FUNDING FACTOR + CRASH/VOL OVERLAY — WALK-FORWARD OOS =====")
    print(f"universe={len(syms)} intervals={T} folds={len(folds)}")
    for fl in fold_log:
        c = fl["chosen"]
        print(f"  fold {fl['fold']}: q={c['quantile']} H={c['hold']} L={c['vol_lookback']} "
              f"tv={c['target_vol']} dd_cut={c['dd_cut']} | train_sh={fl['train_sharpe']:+.2f} "
              f"-> TEST sh={fl['test_sharpe']:+.2f} ann={fl['test_ann_pct']:+.1f}% expo={fl['avg_exposure']} n={fl['test_n']}")
    print("\n  ---- AGGREGATE OUT-OF-SAMPLE ----")
    print(f"  Sharpe        : {agg['sharpe']}")
    print(f"  Ann return    : {agg['ann_return_pct']}%")
    print(f"  Monthly (comp): {agg['monthly_pct']}%")
    print(f"  Max drawdown  : {agg['max_dd_pct']}%")
    print(f"  Win rate      : {agg['win_rate']}%  over n={agg['n']} rebalances")
    print(f"  Avg exposure  : {agg['avg_exposure']} (1.0=full, <1 = de-risked)")
    base_dd, base_sh = -19.1, 0.286   # from xs_funding.py OOS for direct comparison
    print(f"\n  vs base (no overlay): Sharpe {base_sh} -> {agg['sharpe']} | maxDD {base_dd}% -> {agg['max_dd_pct']}%")
    verdict = "REAL EDGE (OOS)" if agg["sharpe"] > 0.8 and agg["ann_return_pct"] > 0 else "NO OOS EDGE"
    print(f"  VERDICT: {verdict}")
    print("\nWrote models_local/xs_funding_overlay_report.json")
    return report


if __name__ == "__main__":
    run()
