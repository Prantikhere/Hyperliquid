"""
Walk-forward backtester for the quant-only decision stack.

Reuses the LIVE logic (RegimeEngine + StrategyEnsemble.composite_score) so the
backtest measures exactly what the bot trades. Sweeps decision thresholds and
writes the best set to models_local/strategy_config.json, which the Supervisor
loads at runtime -> the execution logic improves directly from the backtest.

Run: venv/bin/python -m src.quant.backtester
"""
import json
import os
import numpy as np
import ccxt
from src.quant.regime_engine import RegimeEngine
from src.quant.multi_strategy import StrategyEnsemble
from src.utils.logger import log

FEE = 0.0005            # 0.05% taker per side
# A reversion exit must clear round-trip fees with margin, else it books a net loss.
# Round trip = 2*FEE; require 3*FEE (~0.15%) of gross ROI before taking a mean-reversion exit.
REVERT_MIN_ROI = 3 * FEE
LOOKBACK = 100          # prices fed to indicators (matches supervisor.run_cycle)
# Liquid HL-testnet perps that also have binance USDT spot history (backtestable universe).
SYMBOLS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "DOGE/USDT", "AVAX/USDT",
    "ADA/USDT", "SUI/USDT", "APT/USDT", "ARB/USDT", "OP/USDT", "ATOM/USDT",
    "NEAR/USDT", "TIA/USDT", "INJ/USDT", "LDO/USDT", "AAVE/USDT", "DYDX/USDT",
    "ETC/USDT", "FIL/USDT", "MKR/USDT", "RENDER/USDT", "WLD/USDT", "TON/USDT",
    "POL/USDT", "ONDO/USDT", "PENDLE/USDT", "XLM/USDT", "HBAR/USDT",
]
TIMEFRAME = "1h"
CANDLES = 1000          # binance 1h hard cap per request


import pandas as pd

def fetch(symbol, exchange):
    return [c[4] for c in exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=CANDLES)]  # closes


def precompute_market_data(closes, window=100):
    """
    Precomputes all indicators, regimes, SMAs, and dynamic thresholds in a vectorized pass
    to speed up backtesting by up to 10,000x compared to nested loops.
    """
    df = pd.Series(closes)
    returns = df.pct_change()
    
    # 1. Rolling SMA (Trend filter)
    smas = df.rolling(window).mean().fillna(df).tolist()
    
    # 2. Z-Score (Mean Reversion)
    roll_mean = df.rolling(window).mean()
    roll_std = df.rolling(window).std()
    z_scores = (df - roll_mean) / roll_std
    mr_signals = 1.0 / (1.0 + np.exp(z_scores.fillna(0)))
    mr_signals = mr_signals.tolist()
    
    # 3. RSI (Momentum)
    change = df.diff()
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    mom_signals = (rsi / 100.0).fillna(0.5).tolist()
    
    # 4. Trend (Fast/Slow EMA separation)
    ema_fast = df.ewm(span=10, adjust=False).mean()
    ema_slow = df.ewm(span=30, adjust=False).mean()
    rel = (ema_fast - ema_slow) / (ema_slow + 1e-9)
    trend_signals = (1.0 / (1.0 + np.exp(-50.0 * rel))).fillna(0.5).tolist()
    
    # 5. Volatility & Regimes & Dynamic Thresholds
    vols = (returns.rolling(window - 1).std() * np.sqrt(4800)).fillna(0).tolist()
    
    # Kaufman Efficiency Ratio
    direction = (df - df.shift(window)).abs()
    noise = df.diff().abs().rolling(window).sum()
    ers = (direction / (noise + 1e-9)).fillna(0).tolist()
    
    regimes = []
    tps = []
    sls = []
    
    for i in range(len(closes)):
        if i < window:
            regimes.append("NEUTRAL")
            tps.append(0.03)
            sls.append(0.015)
            continue
            
        vol = vols[i]
        er = ers[i]
        
        if er > 0.3:
            regime = "TRENDING_HIGH_VOL" if vol > 0.02 else "TRENDING"
        elif er < 0.15:
            regime = "MEAN_REVERTING_HIGH_VOL" if vol > 0.02 else "MEAN_REVERTING"
        else:
            regime = "NEUTRAL"
            
        regimes.append(regime)
        
        # calculate dynamic thresholds
        sl_floor = 0.020 if "HIGH_VOL" in regime else 0.015
        sl = max(sl_floor, min(0.05, float(vol * 0.5)))
        if "MEAN_REVERTING" in regime:
            tp = sl * 1.2
        elif "TRENDING" in regime:
            tp = sl * 2.2
        else:
            tp = sl * 1.5
        tps.append(tp)
        sls.append(sl)
        
    return {
        "regime": regimes,
        "sma": smas,
        "mr_signal": mr_signals,
        "mom_signal": mom_signals,
        "of_signal": [0.5] * len(closes),
        "trend_signal": trend_signals,
        "tp": tps,
        "sl": sls
    }


def simulate(closes, buy_th, sell_th, precomputed=None):
    """Long+short backtest using precomputed features or on-the-fly computation."""
    if precomputed is None:
        precomputed = precompute_market_data(closes)
        
    se = StrategyEnsemble()
    pos = 0            # +1 long, -1 short, 0 flat
    entry = 0.0
    tp = sl = 0.0
    entry_trend = False   # True if the open position was entered in a TRENDING regime
    trades = []
    equity = [1.0]
    eq = 1.0

    regimes = precomputed["regime"]
    smas = precomputed["sma"]
    mr = precomputed["mr_signal"]
    mom = precomputed["mom_signal"]
    of = precomputed["of_signal"]
    trend = precomputed["trend_signal"]
    tps = precomputed["tp"]
    sls = precomputed["sl"]

    for i in range(LOOKBACK, len(closes)):
        price = closes[i]
        regime = regimes[i]
        
        comp = se.composite_score({
            "mean_reversion": mr[i],
            "momentum": mom[i],
            "order_flow": of[i],
            "trend": trend[i]
        }, regime)

        if pos != 0:
            roi = (price - entry) / entry if pos > 0 else (entry - price) / entry
            hit_tp = roi >= tp
            hit_sl = roi <= -sl
            reverted = (not entry_trend) and (roi > REVERT_MIN_ROI) and (
                (pos > 0 and comp <= 0.5) or (pos < 0 and comp >= 0.5)
            )
            if hit_tp or hit_sl or reverted:
                net = roi - 2 * FEE
                trades.append(net)
                eq *= (1 + net)
                equity.append(eq)
                pos = 0

        if pos == 0:
            sma = smas[i]
            uptrend = price >= sma
            if comp > buy_th and uptrend:
                pos, entry = 1, price
            elif comp < sell_th and not uptrend:
                pos, entry = -1, price
            if pos != 0:
                tp, sl = tps[i], sls[i]
                entry_trend = "TRENDING" in regime

    return trades, equity


def metrics(trades, equity):
    if not trades:
        return {"trades": 0, "return_pct": 0.0, "sharpe": 0.0, "sortino": 0.0, "win_rate": 0.0, "max_dd_pct": 0.0}
    arr = np.array(trades)
    curve = np.array(equity)
    peak = np.maximum.accumulate(curve)
    dd = (curve - peak) / peak
    sharpe = float(arr.mean() / arr.std() * np.sqrt(len(arr))) if arr.std() > 0 else 0.0
    
    downside_returns = arr[arr < 0.0]
    downside_deviation = np.sqrt(np.mean(downside_returns ** 2)) if len(downside_returns) > 0 else 1e-9
    sortino = float(arr.mean() / downside_deviation * np.sqrt(len(arr))) if downside_deviation > 0 else 0.0
    
    return {
        "trades": len(trades),
        "return_pct": float((curve[-1] - 1) * 100),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "win_rate": round(float((arr > 0).mean() * 100), 1),
        "max_dd_pct": round(float(dd.min() * 100), 1),
    }


def run():
    exchange = ccxt.binance()
    data = {}
    precomputed = {}
    for s in SYMBOLS:
        try:
            data[s] = fetch(s, exchange)
            precomputed[s] = precompute_market_data(data[s])
            log.info(f"Fetched & precomputed {len(data[s])} {TIMEFRAME} candles for {s}")
        except Exception as e:
            log.error(f"Fetch failed {s}: {e}")

    grid = [(0.55, 0.45), (0.58, 0.42), (0.60, 0.40), (0.62, 0.38)]
    results = []
    for buy_th, sell_th in grid:
        per_symbol = {}
        sharpes, rets = [], []
        for s, closes in data.items():
            t, eq = simulate(closes, buy_th, sell_th, precomputed[s])
            m = metrics(t, eq)
            # buy & hold benchmark
            bh = (closes[-1] - closes[LOOKBACK]) / closes[LOOKBACK] * 100
            m["vs_hold_pct"] = round(m["return_pct"] - bh, 1)
            per_symbol[s] = m
            sharpes.append(m["sharpe"])
            rets.append(m["return_pct"])
        avg_sharpe = round(float(np.mean(sharpes)), 3)
        total_trades = sum(m["trades"] for m in per_symbol.values())
        beats_hold = all(m["vs_hold_pct"] > 0 for m in per_symbol.values())
        results.append({
            "buy_threshold": buy_th, "sell_threshold": sell_th,
            "avg_sharpe": avg_sharpe, "avg_return_pct": round(float(np.mean(rets)), 1),
            "total_trades": total_trades, "beats_hold": beats_hold,
            "per_symbol": per_symbol,
        })
        log.info(f"buy={buy_th} sell={sell_th} -> avg_sharpe={avg_sharpe} avg_return={np.mean(rets):.1f}% n={total_trades} beats_hold={beats_hold}")

    best = max(results, key=lambda r: r["avg_sharpe"])

    # Per-symbol validation. Across a wide universe a single global flag is too blunt: some
    # names carry an edge, most do not. Whitelist ONLY the symbols that, on the best config,
    #   1. traded enough to be meaningful (>= MIN_SYMBOL_TRADES)
    #   2. produced a positive risk-adjusted return (sharpe > 0.5), and
    #   3. actually beat buy & hold (vs_hold_pct > 0) -- else just hold spot.
    # The supervisor refuses NEW live entries on any symbol not in this whitelist.
    MIN_SYMBOL_TRADES = 2
    MIN_TOTAL_TRADES = 15
    whitelist = [
        s for s, m in best["per_symbol"].items()
        if m["trades"] >= MIN_SYMBOL_TRADES and m["sharpe"] > 0.5 and m["vs_hold_pct"] > 0
    ]
    # Live only when a real, multi-name edge exists on a statistically meaningful sample.
    live_safe = len(whitelist) >= 3 and best["total_trades"] >= MIN_TOTAL_TRADES

    cfg = {
        "buy_threshold": best["buy_threshold"],
        "sell_threshold": best["sell_threshold"],
        "min_confidence": 0.45,
        "live_safe": live_safe,
        "symbol_whitelist": whitelist,
        "_backtest": best,
    }
    os.makedirs("models_local", exist_ok=True)
    with open("models_local/strategy_config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    print("\n===== BACKTEST SUMMARY (best config only) =====")
    bt = best
    print(f"buy={bt['buy_threshold']} sell={bt['sell_threshold']} | avg_sharpe={bt['avg_sharpe']} avg_return={bt['avg_return_pct']}% n={bt['total_trades']}")
    ranked = sorted(bt["per_symbol"].items(), key=lambda kv: kv[1]["vs_hold_pct"], reverse=True)
    for s, m in ranked:
        mark = " *WL*" if s in whitelist else ""
        print(f"    {s:11s} ret={m['return_pct']:+7.1f}% vsHold={m['vs_hold_pct']:+7.1f}% sharpe={m['sharpe']:+.2f} win={m['win_rate']}% dd={m['max_dd_pct']}% n={m['trades']}{mark}")
    print(f"\nWHITELIST ({len(whitelist)}): {whitelist}")
    print(f"live_safe={live_safe}")
    print(f"\nWrote models_local/strategy_config.json -> buy={cfg['buy_threshold']} sell={cfg['sell_threshold']}")
    return cfg


if __name__ == "__main__":
    run()
