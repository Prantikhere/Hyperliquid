"""
Walk-forward OUT-OF-SAMPLE validation for the quant decision stack.

The plain backtester (backtester.py) sweeps thresholds and selects the whitelist on
the SAME 1000h window it validates on -> in-sample (IS) selection, i.e. data snooping.
A symbol can look great purely because the one window happened to suit it.

This module answers the only question that matters: does the edge survive on data the
selection never saw? It uses rolling walk-forward folds:

    fold: pick the best global (buy,sell) threshold on the TRAIN segment, then trade it
          on the immediately-following TEST segment (never seen during selection).

OOS trades are concatenated across folds per symbol. A symbol is OOS-whitelisted only if
its out-of-sample record clears the same bar the live gate uses (>=5 trades, sharpe>0.5,
beats buy & hold). The IS vs OOS whitelist diff tells you how much of live_safe was a mirage.

Run: venv/bin/python -m src.quant.walkforward
Writes: models_local/walkforward_report.json   (does NOT touch strategy_config.json)
"""
import json
import os
import numpy as np
import ccxt
from src.quant.backtester import simulate, metrics, fetch, SYMBOLS, LOOKBACK, FEE, TIMEFRAME, CANDLES
from src.utils.logger import log

GRID = [(0.55, 0.45), (0.58, 0.42), (0.60, 0.40), (0.62, 0.38)]

# Rolling walk-forward geometry over the 1000-bar series (LOOKBACK=100 warmup per sim).
TRAIN_LEN = 500
TEST_LEN = 150
FIRST_TEST_START = 500          # first test region begins here; train = [start-TRAIN_LEN, start)

# OOS whitelist bar -- identical to the live gate in backtester.run().
MIN_SYMBOL_TRADES = 5
MIN_TOTAL_TRADES = 30


def make_folds(n):
    """Rolling (train, test) index windows tiling the back of the series."""
    folds = []
    ts = FIRST_TEST_START
    while ts + TEST_LEN <= n and ts - TRAIN_LEN >= 0:
        folds.append({
            "train": (ts - TRAIN_LEN, ts),
            "test": (ts, ts + TEST_LEN),
        })
        ts += TEST_LEN
    return folds


def _sim_segment(closes, seg_start, seg_end, buy_th, sell_th):
    """Simulate decisions over [seg_start, seg_end), feeding LOOKBACK bars of prior
    history so the first decision bar has a full window (matches live warmup)."""
    lo = seg_start - LOOKBACK
    if lo < 0:
        return [], [1.0]
    return simulate(closes[lo:seg_end], buy_th, sell_th)


def pick_threshold_on_train(data, tr_start, tr_end):
    """Choose the single global (buy,sell) that maximises avg train sharpe across symbols.
    Mirrors backtester.run() which also selects one global threshold set."""
    best_th, best_score = GRID[0], -1e9
    for buy_th, sell_th in GRID:
        sharpes = []
        for closes in data.values():
            t, eq = _sim_segment(closes, tr_start, tr_end, buy_th, sell_th)
            sharpes.append(metrics(t, eq)["sharpe"])
        score = float(np.mean(sharpes)) if sharpes else -1e9
        if score > best_score:
            best_score, best_th = score, (buy_th, sell_th)
    return best_th, round(best_score, 3)


def run():
    exchange = ccxt.binance()
    data = {}
    for s in SYMBOLS:
        try:
            data[s] = fetch(s, exchange)
        except Exception as e:
            log.error(f"Fetch failed {s}: {e}")
    if not data:
        print("No data fetched. Abort.")
        return

    n = min(len(c) for c in data.values())
    folds = make_folds(n)
    log.info(f"Walk-forward: {len(folds)} folds over {n} bars, {len(data)} symbols")
    if not folds:
        print("Series too short for walk-forward geometry. Abort.")
        return

    # Per-symbol OOS accumulators.
    oos_trades = {s: [] for s in data}          # concatenated net-of-fee trade ROIs (test only)
    oos_hold_factors = {s: [] for s in data}    # (1+hold_ret) per test segment, for compounding
    fold_log = []

    for fi, fold in enumerate(folds):
        tr0, tr1 = fold["train"]
        te0, te1 = fold["test"]
        (buy_th, sell_th), train_score = pick_threshold_on_train(data, tr0, tr1)
        fold_log.append({
            "fold": fi, "train": [tr0, tr1], "test": [te0, te1],
            "chosen_buy": buy_th, "chosen_sell": sell_th, "train_avg_sharpe": train_score,
        })
        log.info(f"fold {fi}: train[{tr0}:{tr1}] -> buy={buy_th} sell={sell_th} (train_sharpe={train_score}); test[{te0}:{te1}]")
        for s, closes in data.items():
            t, eq = _sim_segment(closes, te0, te1, buy_th, sell_th)   # OOS: unseen threshold-on-train
            oos_trades[s].extend(t)
            hold = (closes[te1 - 1] - closes[te0]) / closes[te0]
            oos_hold_factors[s].append(1.0 + hold)

    # Aggregate OOS metrics per symbol (compound test trades in chronological fold order).
    per_symbol = {}
    for s in data:
        tr = oos_trades[s]
        eq = [1.0]
        e = 1.0
        for net in tr:
            e *= (1 + net)
            eq.append(e)
        m = metrics(tr, eq)
        hold_total = (float(np.prod(oos_hold_factors[s])) - 1.0) * 100 if oos_hold_factors[s] else 0.0
        m["hold_pct"] = round(hold_total, 1)
        m["vs_hold_pct"] = round(m["return_pct"] - hold_total, 1)
        per_symbol[s] = m

    oos_whitelist = [
        s for s, m in per_symbol.items()
        if m["trades"] >= MIN_SYMBOL_TRADES and m["sharpe"] > 0.5 and m["vs_hold_pct"] > 0
    ]
    total_oos_trades = sum(m["trades"] for m in per_symbol.values())
    oos_live_safe = len(oos_whitelist) >= 3 and total_oos_trades >= MIN_TOTAL_TRADES
    avg_sharpe = round(float(np.mean([m["sharpe"] for m in per_symbol.values()])), 3)
    avg_return = round(float(np.mean([m["return_pct"] for m in per_symbol.values()])), 1)

    # Load the in-sample whitelist currently driving live, for the honesty diff.
    is_whitelist = []
    try:
        with open("models_local/strategy_config.json") as f:
            is_whitelist = json.load(f).get("symbol_whitelist", [])
    except Exception:
        pass

    survived = [s for s in is_whitelist if s in oos_whitelist]
    mirage = [s for s in is_whitelist if s not in oos_whitelist]
    new_oos = [s for s in oos_whitelist if s not in is_whitelist]

    report = {
        "method": "rolling walk-forward, threshold selected on train, measured on test",
        "folds": fold_log,
        "n_folds": len(folds),
        "oos_avg_sharpe": avg_sharpe,
        "oos_avg_return_pct": avg_return,
        "oos_total_trades": total_oos_trades,
        "oos_whitelist": oos_whitelist,
        "oos_live_safe": oos_live_safe,
        "in_sample_whitelist": is_whitelist,
        "survived_oos": survived,
        "in_sample_mirage": mirage,
        "new_in_oos": new_oos,
        "per_symbol_oos": per_symbol,
    }
    os.makedirs("models_local", exist_ok=True)
    with open("models_local/walkforward_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("\n===== WALK-FORWARD OUT-OF-SAMPLE REPORT =====")
    print(f"{len(folds)} folds | OOS avg_sharpe={avg_sharpe} avg_return={avg_return}% n={total_oos_trades}")
    for fl in fold_log:
        print(f"  fold {fl['fold']}: train{fl['train']} pick buy={fl['chosen_buy']}/sell={fl['chosen_sell']} "
              f"(train_sharpe={fl['train_avg_sharpe']}) -> test{fl['test']}")
    print("\n  -- per-symbol OUT-OF-SAMPLE (sorted by vsHold) --")
    ranked = sorted(per_symbol.items(), key=lambda kv: kv[1]["vs_hold_pct"], reverse=True)
    for s, m in ranked:
        mark = " *OOS-WL*" if s in oos_whitelist else ""
        print(f"    {s:11s} ret={m['return_pct']:+7.1f}% vsHold={m['vs_hold_pct']:+7.1f}% "
              f"sharpe={m['sharpe']:+.2f} win={m['win_rate']}% n={m['trades']}{mark}")
    print(f"\n  IN-SAMPLE whitelist  ({len(is_whitelist)}): {is_whitelist}")
    print(f"  OUT-OF-SAMPLE wlist  ({len(oos_whitelist)}): {oos_whitelist}")
    print(f"  SURVIVED OOS         ({len(survived)}): {survived}")
    print(f"  IN-SAMPLE MIRAGE     ({len(mirage)}): {mirage}   <- looked good IS, failed OOS")
    print(f"  NEW IN OOS           ({len(new_oos)}): {new_oos}")
    print(f"\n  OOS live_safe = {oos_live_safe}   (IS live_safe was True)")
    print("\nWrote models_local/walkforward_report.json (strategy_config.json untouched)")
    return report


if __name__ == "__main__":
    run()
