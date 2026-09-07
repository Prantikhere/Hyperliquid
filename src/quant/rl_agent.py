"""
RL trading agent -- TensorTrade-aligned pipeline built on the existing hand-rolled
TradingEnv/RewardScheme (src/quant/env.py, src/quant/reward_scheme.py) instead of the
actual `tensortrade` package, which is unmaintained (last release 2021, pinned to
pre-gymnasium `gym` + numpy<2) and will not install cleanly on this venv's Python 3.14.

Data source: Hyperliquid TESTNET (same venue/sandbox as the live executors), across
every tradable USDC-margined perp swap market -- not Binance. HL testnet 1h history is
capped and fluctuates (~150-200h observed, see pairs_arb_executor.py's LOOKBACK_H/
MIN_CANDLES notes), so each symbol trains/evaluates on however much history it has
above MIN_CANDLES, not a fixed window.

Pipeline, matching the PROMOTE-CANDIDATE gate already used by pairs_stat_arb.py /
walkforward.py (>=15 trades, sortino>=0.5):

  1. DATA:   pull every real USDC swap market from HL testnet (load_markets()),
             drop obvious test/junk listings, fetch 1h OHLCV per symbol.
  2. SPLIT:  chronological (no shuffling) train/val/test, walk-forward style --
             train on the older segment, evaluate out-of-sample on the newer one.
  3. CONFIG: single dict below, no YAML/CLI framework needed at this scale.
  4. TRAIN:  stable-baselines3 PPO on a TradingGymEnv wrapping the train segment.
  5. EVAL:   deterministic rollout on the held-out val/test segments; compute
             Sortino (via RiskAdjustedReturns, matches live position-sizing) + trade count.
  6. REWARD: only agents clearing the promote gate get their model + report saved to
             models_local/ -- failing agents are discarded, exactly like the other
             quant models never make it into strategy_config.json.

Ray: intentionally NOT used. Ray/RLlib pays off when training many symbols/seeds truly
in parallel; this sweep runs sequentially against one HL connection (rate-limit
friendly) and each symbol only trains for a few seconds to a couple minutes -- adding
a Ray cluster here is pure ops overhead for no wall-clock win. Revisit if this needs to
fan out across machines.

Run (single symbol):  venv/bin/python -m src.quant.rl_agent ETC/USDC:USDC
Run (full universe):  venv/bin/python -m src.quant.rl_agent --universe
Writes: models_local/rl_agent_<symbol>.zip + _report.json per promoted symbol,
        models_local/rl_agent_universe_report.json summary for --universe runs.
"""
import json
import os
import re
import sys
import numpy as np
import pandas as pd
import ccxt
from dotenv import load_dotenv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from src.quant.gym_env import TradingGymEnv
from src.quant.reward_scheme import RiskAdjustedReturns
from src.utils.logger import log

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

FEE = 0.0005
TIMEFRAME = "1h"
CANDLES = 300            # requested; HL testnet caps/truncates silently, we use whatever comes back
MIN_CANDLES = 120        # floor below which a symbol's history can't support a meaningful split

CONFIG = {
    "train_frac": 0.70,
    "val_frac": 0.15,       # remainder (0.15) is held out as test, never touched during model selection
    "total_timesteps": 20_000,          # single-symbol run
    "total_timesteps_sweep": 8_000,     # per-symbol budget when sweeping the whole universe
    "policy": "MlpPolicy",
    "seed": 42,
    "promote_min_trades": 15,
    "promote_min_sortino": 0.5,
}

MODELS_DIR = "models_local"

# HL testnet lists throwaway/staging markets alongside real ones (e.g. "ANGRY-TEST").
_JUNK_RE = re.compile(r"TEST|^\d|-\d+$", re.IGNORECASE)


def hl_client():
    hl = ccxt.hyperliquid({
        "privateKey": os.getenv("HL_PRIVATE") or os.getenv("HL_PRIVATE_KEY"),
        "walletAddress": os.getenv("HL_WALLET_ADDRESS"),
        "options": {"defaultType": "swap"},
    })
    hl.set_sandbox_mode(True)
    return hl


def hl_universe(hl):
    markets = hl.load_markets()
    symbols = [
        m["symbol"] for m in markets.values()
        if m.get("swap") and m.get("quote") == "USDC" and not _JUNK_RE.search(m["base"])
    ]
    return sorted(symbols)


def fetch_hl_closes(hl, symbol, limit=CANDLES):
    candles = hl.fetch_ohlcv(symbol, TIMEFRAME, limit=limit)
    return [c[4] for c in candles]


def chronological_split(closes, train_frac, val_frac):
    n = len(closes)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)
    return closes[:train_end], closes[train_end:val_end], closes[val_end:]


def rollout(model, prices, fee, deterministic=True):
    """Deterministic policy rollout on a held-out segment. Returns (returns, n_trades)."""
    env = TradingGymEnv(prices, fee=fee, reward_scheme=RiskAdjustedReturns("sortino"))
    obs, _ = env.reset()
    n_trades = 0
    last_pos = 0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        if info["position"] != last_pos:
            n_trades += 1
            last_pos = info["position"]
        done = terminated or truncated
    return env.env.returns, n_trades


def evaluate(returns, n_trades):
    scheme = RiskAdjustedReturns("sortino")
    sortino = scheme.get_reward(pd.Series(returns))
    total_roi = float(np.prod([1 + r for r in returns]) - 1) if returns else 0.0
    return {"sortino": sortino, "total_roi": total_roi, "n_trades": n_trades}


def train_and_gate(symbol, hl, total_timesteps=None):
    total_timesteps = total_timesteps or CONFIG["total_timesteps"]
    closes = fetch_hl_closes(hl, symbol)
    if len(closes) < MIN_CANDLES:
        log.warning(f"[RL_AGENT] {symbol}: only {len(closes)} candles (<{MIN_CANDLES}), skipping.")
        return {"symbol": symbol, "skipped": True, "n_candles": len(closes)}
    log.info(f"[RL_AGENT] {symbol}: fetched {len(closes)} {TIMEFRAME} candles")

    train, val, test = chronological_split(closes, CONFIG["train_frac"], CONFIG["val_frac"])
    if len(train) < 20 or len(val) < 5:
        log.warning(f"[RL_AGENT] {symbol}: split too small (train={len(train)} val={len(val)}), skipping.")
        return {"symbol": symbol, "skipped": True, "n_candles": len(closes)}
    log.info(f"[RL_AGENT] {symbol} split train={len(train)} val={len(val)} test={len(test)}")

    train_env = Monitor(TradingGymEnv(train, fee=FEE, reward_scheme=RiskAdjustedReturns("sortino")))
    model = PPO(CONFIG["policy"], train_env, seed=CONFIG["seed"], verbose=0)
    model.learn(total_timesteps=total_timesteps)

    val_returns, val_trades = rollout(model, val, FEE)
    val_metrics = evaluate(val_returns, val_trades)
    log.info(f"[RL_AGENT] {symbol} VAL sortino={val_metrics['sortino']:.2f} "
             f"roi={val_metrics['total_roi']*100:.2f}% trades={val_metrics['n_trades']}")

    promoted = (val_metrics["n_trades"] >= CONFIG["promote_min_trades"]
                and val_metrics["sortino"] >= CONFIG["promote_min_sortino"])

    report = {"symbol": symbol, "n_candles": len(closes), "val": val_metrics, "promoted": promoted}

    safe_sym = symbol.replace("/", "_").replace(":", "_")
    os.makedirs(MODELS_DIR, exist_ok=True)
    if promoted:
        test_returns, test_trades = rollout(model, test, FEE)
        report["test"] = evaluate(test_returns, test_trades)
        model.save(os.path.join(MODELS_DIR, f"rl_agent_{safe_sym}"))
        log.info(f"[RL_AGENT] {symbol} PROMOTED -- test sortino={report['test']['sortino']:.2f} "
                 f"roi={report['test']['total_roi']*100:.2f}%. Model saved.")
    else:
        log.info(f"[RL_AGENT] {symbol} NOT promoted (gate: >={CONFIG['promote_min_trades']} trades, "
                 f"sortino>={CONFIG['promote_min_sortino']}). Model discarded.")

    with open(os.path.join(MODELS_DIR, f"rl_agent_{safe_sym}_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    return report


def train_universe():
    hl = hl_client()
    symbols = hl_universe(hl)
    log.info(f"[RL_AGENT] universe sweep: {len(symbols)} HL testnet USDC perp markets")

    results = []
    for i, symbol in enumerate(symbols, 1):
        log.info(f"[RL_AGENT] ({i}/{len(symbols)}) {symbol}")
        try:
            report = train_and_gate(symbol, hl, total_timesteps=CONFIG["total_timesteps_sweep"])
        except Exception as e:
            log.error(f"[RL_AGENT] {symbol} failed: {e}")
            report = {"symbol": symbol, "error": str(e)}
        results.append(report)

    promoted = [r for r in results if r.get("promoted")]
    summary = {
        "n_symbols": len(symbols),
        "n_promoted": len(promoted),
        "promoted_symbols": [r["symbol"] for r in promoted],
        "results": results,
    }
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, "rl_agent_universe_report.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"[RL_AGENT] universe sweep done: {len(promoted)}/{len(symbols)} promoted -> {promoted and summary['promoted_symbols']}")
    return summary


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--universe":
        train_universe()
    else:
        sym = sys.argv[1] if len(sys.argv) > 1 else "ETC/USDC:USDC"
        train_and_gate(sym, hl_client())
