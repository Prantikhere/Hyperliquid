"""
Cross-sectional FUNDING factor — perp-only, dollar-neutral, walk-forward OOS.

Funding carry (funding_carry.py) was the one edge that survived OOS, but it needs a spot leg
to be truly delta-neutral (impossible on HL testnet). This is its perp-only cousin and does not
need spot at all:

  Each funding interval, rank the universe by funding rate.
    - SHORT perp on the richest-funding coins  -> you RECEIVE their (positive) funding.
    - LONG  perp on the most-negative-funding coins -> longs RECEIVE funding there too.
  Dollar-neutral (equal long/short notional) => market beta ~= 0.

Return per interval = funding harvested  +  price spread (loser basket - winner basket).
There is a second, documented tailwind: extreme positive funding marks over-leveraged longs
that tend to MEAN-REVERT, so shorting them earns funding AND reversal. Pure market-neutral,
pure structure -- no directional prediction.

Honesty: quantile/rebalance params chosen on TRAIN funding+price only, scored on unseen TEST.
Costs: taker fee per leg * turnover each rebalance. Funding sign handled explicitly.

Run: venv/bin/python -m src.quant.xs_funding
Writes: models_local/xs_funding_report.json
"""
import json
import os
import time
import numpy as np
import ccxt
from src.quant.backtester import SYMBOLS
from src.quant.funding_carry import fetch_funding, INTERVALS_PER_YEAR
from src.utils.logger import log

FEE = 0.0005                    # 5 bps taker per side
TF = "8h"                       # binance funding cadence
HISTORY_YEARS = 3

# Walk-forward geometry (in 8h intervals; 3/day).
TRAIN_LEN = 800                 # ~266 days
TEST_LEN = 300                  # ~100 days
FIRST_TEST_START = 800

# Params searched on TRAIN. hold = intervals between rebalances (1 = every funding stamp).
GRID = []
for q in (0.2, 0.3, 0.4):
    for hold in (1, 3):
        GRID.append({"quantile": q, "hold": hold})


def fetch_price_8h(symbol, exchange, years=HISTORY_YEARS):
    """Paginate 8h closes ~years back. Returns dict ts_ms->close."""
    market = symbol.replace("/USDT", "/USDT:USDT")
    since = exchange.milliseconds() - int(years * 365 * 24 * 3600 * 1000)
    now = exchange.milliseconds()
    out = {}
    step = 8 * 3600 * 1000
    while since < now:
        batch = exchange.fetch_ohlcv(market, TF, since=since, limit=1000)
        if not batch:
            break
        for r in batch:
            out[r[0]] = r[4]
        last = batch[-1][0]
        if last + step <= since:
            break
        since = last + step
        if len(batch) < 2:
            break
        time.sleep(exchange.rateLimit / 1000.0)
    return out


def fetch_funding_ts(symbol, exchange):
    """Funding as dict ts_ms->rate (fetch_funding returns values only; re-fetch with stamps)."""
    market = symbol.replace("/USDT", "/USDT:USDT")
    since = exchange.milliseconds() - int(HISTORY_YEARS * 365 * 24 * 3600 * 1000)
    now = exchange.milliseconds()
    out = {}
    while since < now:
        batch = exchange.fetch_funding_rate_history(market, since=since, limit=1000)
        if not batch:
            break
        for h in batch:
            ts, fr = h.get("timestamp"), h.get("fundingRate")
            if ts is not None and fr is not None:
                out[ts] = float(fr)
        last = batch[-1].get("timestamp")
        if last is None or last + 1 <= since:
            break
        since = last + 1
        if len(batch) < 2:
            break
        time.sleep(exchange.rateLimit / 1000.0)
    return out


BUCKET = 8 * 3600 * 1000        # 8h in ms

def build(data):
    """Align funding + 8h price by flooring both to 8h buckets (funding stamps and price-bar
    opens can differ by seconds/offset; exact ms join misses). Returns syms, buckets, fund, px."""
    def b(ts):
        return ts // BUCKET
    common = set()
    for fmap, _ in data.values():
        common |= {b(t) for t in fmap}
    buckets = sorted(common)
    syms = list(data.keys())
    T, N = len(buckets), len(syms)
    idx = {bk: i for i, bk in enumerate(buckets)}
    fund = np.full((T, N), np.nan)
    px = np.full((T, N), np.nan)
    for j, s in enumerate(syms):
        fmap, pmap = data[s]
        for t, r in fmap.items():
            if b(t) in idx:
                fund[idx[b(t)], j] = r
        for t, c in pmap.items():
            if b(t) in idx:
                px[idx[b(t)], j] = c
    return syms, buckets, fund, px


def simulate(fund, px, cfg, s0, s1):
    """Dollar-neutral funding-ranked long-short over [s0, s1). Returns per-rebalance net rets."""
    q, H = cfg["quantile"], cfg["hold"]
    T, N = fund.shape
    rets = []
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
        order = v[np.argsort(fr[v])]              # ascending funding
        k = max(1, int(len(order) * q))
        longs = order[:k]                          # most-negative funding -> long (receive funding)
        shorts = order[-k:]                        # richest funding -> short (receive funding)
        w = np.zeros(N)
        w[longs] = 0.5 / k
        w[shorts] = -0.5 / k
        price_ret = np.zeros(N)
        price_ret[valid] = fut[valid] / now[valid] - 1.0
        # funding pnl over the H intervals held: short receives +f, long pays. sum f over window.
        fsum = np.zeros(N)
        for hh in range(H):
            row = fund[t + hh, :]
            fsum[valid] += np.where(np.isfinite(row[valid]), row[valid], 0.0)
        r = float(np.nansum(w[valid] * price_ret[valid]) - np.nansum(w[valid] * fsum[valid]))
        r -= float(np.abs(w - prev_w).sum()) * FEE
        rets.append(r)
        prev_w = w
        t += H
    return rets


def perf(rets, hold):
    if not rets:
        return {"n": 0, "sharpe": 0.0, "ann_return_pct": 0.0, "monthly_pct": 0.0, "max_dd_pct": 0.0, "win_rate": 0.0}
    a = np.array(rets)
    eq = np.cumprod(1 + a)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    per_year = (INTERVALS_PER_YEAR / hold)
    mean, sd = a.mean(), a.std()
    sharpe = float(mean / sd * np.sqrt(per_year)) if sd > 0 else 0.0
    ann = float(eq[-1] ** (per_year / len(a)) - 1) if eq[-1] > 0 else -1.0
    return {
        "n": len(a), "sharpe": round(sharpe, 3),
        "ann_return_pct": round(ann * 100, 1),
        "monthly_pct": round(((1 + ann) ** (1 / 12) - 1) * 100, 2) if ann > -1 else -100.0,
        "max_dd_pct": round(float(dd.min()) * 100, 1),
        "win_rate": round(float((a > 0).mean()) * 100, 1),
    }


def make_folds(T):
    folds, ts = [], FIRST_TEST_START
    while ts + TEST_LEN <= T and ts - TRAIN_LEN >= 0:
        folds.append({"train": (ts - TRAIN_LEN, ts), "test": (ts, ts + TEST_LEN)})
        ts += TEST_LEN
    return folds


def pick_cfg(fund, px, tr0, tr1):
    best, best_s = GRID[0], -1e9
    for cfg in GRID:
        m = perf(simulate(fund, px, cfg, tr0, tr1), cfg["hold"])
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
                log.info(f"XSF {s}: {len(fmap)} funding, {len(pmap)} price")
        except Exception as e:
            log.error(f"XSF fetch {s}: {e}")
    if len(data) < 8:
        print(f"Only {len(data)} symbols; need >=8. Abort.")
        return

    syms, ts, fund, px = build(data)
    T = fund.shape[0]
    folds = make_folds(T)
    log.info(f"XSF: {len(syms)} symbols, {T} intervals, {len(folds)} folds")
    if not folds:
        print("Series too short. Abort.")
        return

    oos_rets, oos_holds, fold_log = [], [], []
    for fi, fold in enumerate(folds):
        tr0, tr1 = fold["train"]
        te0, te1 = fold["test"]
        cfg, train_s = pick_cfg(fund, px, tr0, tr1)
        te = simulate(fund, px, cfg, te0, te1)
        oos_rets.extend(te)
        oos_holds.extend([cfg["hold"]] * len(te))
        m = perf(te, cfg["hold"])
        fold_log.append({"fold": fi, "chosen": cfg, "train_sharpe": train_s,
                         "test_sharpe": m["sharpe"], "test_ann_pct": m["ann_return_pct"], "test_n": m["n"]})
        log.info(f"fold {fi}: {cfg} train_sh={train_s} -> test_sh={m['sharpe']} ann={m['ann_return_pct']}%")

    med_hold = int(np.median(oos_holds)) if oos_holds else 1
    agg = perf(oos_rets, med_hold)
    report = {"method": "cross-sectional funding factor, dollar-neutral perp long-short, walk-forward OOS",
              "universe_size": len(syms), "intervals": T, "n_folds": len(folds),
              "oos": agg, "median_hold_intervals": med_hold, "folds": fold_log}
    os.makedirs("models_local", exist_ok=True)
    with open("models_local/xs_funding_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n===== CROSS-SECTIONAL FUNDING FACTOR — WALK-FORWARD OOS =====")
    print(f"universe={len(syms)} intervals={T} folds={len(folds)}")
    for fl in fold_log:
        c = fl["chosen"]
        print(f"  fold {fl['fold']}: q={c['quantile']} H={c['hold']} | train_sh={fl['train_sharpe']:+.2f} "
              f"-> TEST sh={fl['test_sharpe']:+.2f} ann={fl['test_ann_pct']:+.1f}% n={fl['test_n']}")
    print("\n  ---- AGGREGATE OUT-OF-SAMPLE ----")
    print(f"  Sharpe        : {agg['sharpe']}")
    print(f"  Ann return    : {agg['ann_return_pct']}%")
    print(f"  Monthly (comp): {agg['monthly_pct']}%")
    print(f"  Max drawdown  : {agg['max_dd_pct']}%")
    print(f"  Win rate      : {agg['win_rate']}%  over n={agg['n']} rebalances")
    verdict = "REAL EDGE (OOS)" if agg["sharpe"] > 0.8 and agg["ann_return_pct"] > 0 else "NO OOS EDGE"
    print(f"\n  VERDICT: {verdict}")
    print("\nWrote models_local/xs_funding_report.json")
    return report


if __name__ == "__main__":
    run()
