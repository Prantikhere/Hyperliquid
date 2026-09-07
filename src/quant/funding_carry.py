"""
Funding-carry (cash-and-carry) walk-forward backtest.

Unlike the price-technical stack (no OOS edge), funding carry is a STRUCTURAL edge:
in a perp market, longs pay shorts when funding is positive. A delta-neutral
cash-and-carry -- LONG spot + SHORT perp, equal notional -- cancels price risk and
harvests the funding stream. PnL per funding interval:

    + funding_rate            (short perp RECEIVES funding when funding > 0)
    - basis drift             (~0 over time; perp converges to spot -- ignored, delta-neutral)
    - fees on 4 legs per round trip (open spot+perp, close spot+perp) = 4*FEE

So the position is profitable whenever cumulative funding collected exceeds ~4*FEE.
The only decision is WHEN to hold it. We gate on annualized funding: enter when rich,
exit when it decays -- because during NEGATIVE funding the carry PAYS instead of earns.

Walk-forward: entry/exit thresholds chosen on TRAIN funding, applied to unseen TEST.
Benchmark = always-in carry (enter once, hold whole test, 1 round trip).

Data: binance USDT-perp funding history (8h intervals, ~1000 pts ~= 330 days).
Run: venv/bin/python -m src.quant.funding_carry
Writes: models_local/funding_carry_report.json  (touches nothing live)
"""
import json
import os
import numpy as np
import ccxt
from src.quant.backtester import SYMBOLS, FEE
from src.utils.logger import log

# --- realistic cost model (all conservative) ---
FUNDING_FEE = 0.0003        # taker fee per leg (spot + perp legs)
SLIPPAGE = 0.0002           # execution slippage per leg
LEG_COST = FUNDING_FEE + SLIPPAGE
ROUND_TRIP_FEE = 4 * LEG_COST      # open spot+perp, close spot+perp = 4 legs
CAPITAL_COST_ANN = 0.03            # ~3%/yr cost of capital tied up in the delta-neutral hedge
INTERVALS_PER_YEAR = 3 * 365       # binance funding = every 8h
HOLDING_COST = CAPITAL_COST_ANN / INTERVALS_PER_YEAR   # charged every interval a position is held

# Exit is HOLD-THROUGH: only close after funding stays weak for EXIT_PERSIST consecutive
# intervals (dodges bear regimes) instead of oscillating on every dip (which churned fees).
EXIT_PERSIST = 3                   # 3 * 8h = 1 day of sustained weak funding before exit

FUND_LIMIT = 1000
HISTORY_YEARS = 3                   # paginate this far back for bull/bear regime coverage

# Rolling walk-forward over the (now multi-year) funding series.
TRAIN_LEN = 800                    # ~266 days train per fold
TEST_LEN = 300                     # ~100 days OOS per fold
FIRST_TEST_START = 800

# Annualized-funding thresholds to sweep (enter when annual funding >= entry, exit when < exit).
GRID = [(0.05, 0.00), (0.10, 0.02), (0.15, 0.05), (0.20, 0.08)]

MIN_SYMBOL_EPISODES = 2
MIN_TOTAL_EPISODES = 15


def fetch_funding(symbol, exchange):
    """Return chronological per-interval funding rates, paginated ~HISTORY_YEARS back.

    Binance returns at most ~1000 rows per call, so walk `since` forward from
    (now - HISTORY_YEARS) until we reach the present. Deduped and sorted by timestamp."""
    import time
    market = symbol.replace("/USDT", "/USDT:USDT")   # binance linear perp
    since = exchange.milliseconds() - int(HISTORY_YEARS * 365 * 24 * 3600 * 1000)
    now = exchange.milliseconds()
    seen = {}
    while since < now:
        batch = exchange.fetch_funding_rate_history(market, since=since, limit=FUND_LIMIT)
        if not batch:
            break
        for h in batch:
            ts = h.get("timestamp")
            fr = h.get("fundingRate")
            if ts is not None and fr is not None:
                seen[ts] = float(fr)
        last = batch[-1].get("timestamp")
        if last is None or last + 1 <= since:
            break
        since = last + 1
        if len(batch) < 2:
            break
        time.sleep(exchange.rateLimit / 1000.0)
    return [seen[k] for k in sorted(seen)]


def simulate_carry(funding, entry_ann, exit_ann):
    """Hold-through cash-and-carry with realistic costs, over a funding slice.

    Enter when annualized funding >= entry_ann. Then HOLD, collecting funding each interval
    minus a per-interval holding cost, and only exit after funding stays below exit_ann for
    EXIT_PERSIST consecutive intervals (sustained-weakness exit). This avoids the fee churn
    of exiting on every momentary dip. Returns per-episode net returns and intervals held."""
    episodes = []
    in_pos = False
    ep_pnl = 0.0
    weak_streak = 0
    held = 0
    for f in funding:
        ann = f * INTERVALS_PER_YEAR
        if not in_pos and ann >= entry_ann:
            in_pos = True
            ep_pnl = -ROUND_TRIP_FEE * 0.5   # open legs (2 of 4)
            weak_streak = 0
        if in_pos:
            ep_pnl += f - HOLDING_COST       # collect funding, pay capital cost this interval
            held += 1
            weak_streak = weak_streak + 1 if ann < exit_ann else 0
            if weak_streak >= EXIT_PERSIST:
                ep_pnl += -ROUND_TRIP_FEE * 0.5   # close legs (2 of 4)
                episodes.append(ep_pnl)
                in_pos = False
                ep_pnl = 0.0
                weak_streak = 0
    if in_pos:                                # close out at series end
        ep_pnl += -ROUND_TRIP_FEE * 0.5
        episodes.append(ep_pnl)
    return episodes, held


def carry_metrics(episodes, n_intervals):
    if not episodes:
        return {"episodes": 0, "return_pct": 0.0, "sharpe": 0.0, "win_rate": 0.0, "ann_pct": 0.0}
    arr = np.array(episodes)
    total = float(np.prod(1 + arr) - 1)
    sharpe = float(arr.mean() / arr.std() * np.sqrt(len(arr))) if arr.std() > 0 else 0.0
    ann = total * (INTERVALS_PER_YEAR / n_intervals) if n_intervals else 0.0
    return {
        "episodes": len(episodes),
        "return_pct": round(total * 100, 2),
        "sharpe": round(sharpe, 3),
        "win_rate": round(float((arr > 0).mean() * 100), 1),
        "ann_pct": round(ann * 100, 1),
    }


def make_folds(n):
    folds, ts = [], FIRST_TEST_START
    while ts + TEST_LEN <= n and ts - TRAIN_LEN >= 0:
        folds.append({"train": (ts - TRAIN_LEN, ts), "test": (ts, ts + TEST_LEN)})
        ts += TEST_LEN
    return folds


def pick_threshold(data, tr0, tr1):
    best_th, best_score = GRID[0], -1e9
    for entry_ann, exit_ann in GRID:
        sharpes = []
        for f in data.values():
            ep, _ = simulate_carry(f[tr0:tr1], entry_ann, exit_ann)
            sharpes.append(carry_metrics(ep, tr1 - tr0)["sharpe"])
        score = float(np.mean(sharpes)) if sharpes else -1e9
        if score > best_score:
            best_score, best_th = score, (entry_ann, exit_ann)
    return best_th, round(best_score, 3)


def run():
    exchange = ccxt.binance({"options": {"defaultType": "future"}})
    data = {}
    min_needed = FIRST_TEST_START + TEST_LEN   # need at least one full fold
    dropped = []
    for s in SYMBOLS:
        try:
            fr = fetch_funding(s, exchange)
            if len(fr) >= min_needed:
                data[s] = fr
                log.info(f"Funding {s}: {len(fr)} intervals, mean_ann={np.mean(fr)*INTERVALS_PER_YEAR*100:.1f}%")
            else:
                dropped.append((s, len(fr)))
        except Exception as e:
            log.error(f"Funding fetch failed {s}: {e}")
    if dropped:
        log.info(f"Dropped {len(dropped)} short-history symbols: {dropped}")
    if not data:
        print("No funding data. Abort.")
        return

    n = min(len(v) for v in data.values())
    folds = make_folds(n)
    log.info(f"Funding carry walk-forward: {len(folds)} folds over {n} intervals, {len(data)} symbols")
    if not folds:
        print("Funding series too short. Abort.")
        return

    oos_ep = {s: [] for s in data}
    oos_intervals = {s: 0 for s in data}
    alwaysin_ep = {s: [] for s in data}
    fold_log = []

    for fi, fold in enumerate(folds):
        tr0, tr1 = fold["train"]
        te0, te1 = fold["test"]
        (entry_ann, exit_ann), tscore = pick_threshold(data, tr0, tr1)
        fold_log.append({"fold": fi, "train": [tr0, tr1], "test": [te0, te1],
                         "entry_ann": entry_ann, "exit_ann": exit_ann, "train_sharpe": tscore})
        for s, f in data.items():
            ep, held = simulate_carry(f[te0:te1], entry_ann, exit_ann)          # OOS gated
            oos_ep[s].extend(ep)
            oos_intervals[s] += (te1 - te0)
            aep, _ = simulate_carry(f[te0:te1], -1e9, -2e9)                       # always-in benchmark
            alwaysin_ep[s].extend(aep)

    per_symbol = {}
    for s in data:
        m = carry_metrics(oos_ep[s], oos_intervals[s])
        b = carry_metrics(alwaysin_ep[s], oos_intervals[s])
        m["alwaysin_return_pct"] = b["return_pct"]
        m["edge_vs_alwaysin"] = round(m["return_pct"] - b["return_pct"], 2)
        per_symbol[s] = m

    profitable = [s for s, m in per_symbol.items()
                  if m["episodes"] >= MIN_SYMBOL_EPISODES and m["return_pct"] > 0 and m["sharpe"] > 0.5]
    total_ep = sum(m["episodes"] for m in per_symbol.values())
    carry_live_safe = len(profitable) >= 3 and total_ep >= MIN_TOTAL_EPISODES
    avg_ret = round(float(np.mean([m["return_pct"] for m in per_symbol.values()])), 2)
    avg_sharpe = round(float(np.mean([m["sharpe"] for m in per_symbol.values()])), 3)
    avg_ann = round(float(np.mean([m["ann_pct"] for m in per_symbol.values()])), 1)

    report = {
        "method": "delta-neutral cash-and-carry, threshold-gated, walk-forward OOS",
        "fee_round_trip_pct": ROUND_TRIP_FEE * 100,
        "n_folds": len(folds), "folds": fold_log,
        "oos_avg_return_pct": avg_ret, "oos_avg_sharpe": avg_sharpe, "oos_avg_ann_pct": avg_ann,
        "oos_total_episodes": total_ep,
        "carry_profitable_symbols": profitable, "carry_live_safe": carry_live_safe,
        "per_symbol_oos": per_symbol,
    }
    os.makedirs("models_local", exist_ok=True)
    with open("models_local/funding_carry_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n===== FUNDING-CARRY WALK-FORWARD (OOS) =====")
    print(f"delta-neutral cash&carry | round-trip cost={ROUND_TRIP_FEE*100:.2f}% "
          f"(fee+slip {LEG_COST*100:.2f}%/leg x4) | capital {CAPITAL_COST_ANN*100:.0f}%/yr | "
          f"hold-through exit={EXIT_PERSIST}x | {len(folds)} folds")
    print(f"OOS avg_return={avg_ret}% avg_sharpe={avg_sharpe} avg_annualized={avg_ann}% episodes={total_ep}")
    for fl in fold_log:
        print(f"  fold {fl['fold']}: train{fl['train']} pick entry>={fl['entry_ann']*100:.0f}%/exit<{fl['exit_ann']*100:.0f}% "
              f"(train_sharpe={fl['train_sharpe']}) -> test{fl['test']}")
    print("\n  -- per-symbol OOS carry (sorted by return) --")
    ranked = sorted(per_symbol.items(), key=lambda kv: kv[1]["return_pct"], reverse=True)
    for s, m in ranked:
        mark = " *CARRY-WL*" if s in profitable else ""
        print(f"    {s:11s} ret={m['return_pct']:+6.2f}% ann={m['ann_pct']:+6.1f}% sharpe={m['sharpe']:+.2f} "
              f"win={m['win_rate']}% ep={m['episodes']} vsAlwaysIn={m['edge_vs_alwaysin']:+.2f}%{mark}")
    print(f"\n  CARRY-PROFITABLE ({len(profitable)}): {profitable}")
    print(f"  carry_live_safe = {carry_live_safe}")
    print("\nWrote models_local/funding_carry_report.json")
    return report


if __name__ == "__main__":
    run()
