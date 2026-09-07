"""
Cointegrated pairs stat-arb executor for Hyperliquid TESTNET perps. Trades the SPREAD
between two cointegrated assets (market-neutral on the pair), a different alpha source
from hl_perp_ls (cross-sectional momentum) and carry_executor (funding carry).

Validated walk-forward on 4000h Binance history (src/quant/pairs_stat_arb.py,
models_local/pairs_stat_arb_report.json): PROMOTE-CANDIDATE, 8/19 cointegrated pairs
clear sharpe>0.5 on >=15 trades. Of those 8, 7 collide with symbols already traded live by
hl_perp_ls.py (UNIVERSE: BNB,DOGE,AVAX,ARB,OP,NEAR,ADA,APT,ATOM,INJ) or carry_executor.py
(CARRY_RESERVED: BTC,ETH,SOL) -- same wallet, cross-strategy position stomping risk (see
hl_perp_ls.py's own CARRY_RESERVED carve-out). Only ETC/FIL is symbol-clean, so it is the
sole pair traded here:
  ETC/FIL  sharpe=+1.69 win=54.1% dd=-10.7% n=37

Method: rolling LOOKBACK_H-hour OLS hedge ratio (a = hedge*b + c), z-score the spread,
enter |z|>ENTRY_Z, exit |z|<EXIT_Z or stale (MAX_HOLD_H) or flip. Dollar-neutral per pair,
low leverage (a hedge, not a directional bet).

SAFETY: testnet only (set_sandbox_mode). Dry-run by default; PAIRS_ARB_LIVE=yes to arm.
Persistent DD kill-switch (survives restarts, same pattern as hl_perp_ls).

Run (dry):  venv/bin/python -m src.execution.pairs_arb_executor
Run (live): PAIRS_ARB_LIVE=yes venv/bin/python -m src.execution.pairs_arb_executor
"""
import asyncio
import os
import time
import numpy as np
import ccxt.async_support as ccxt
from dotenv import load_dotenv
from src.utils.logger import log
from src.utils.db import DatabaseManager
from src.utils.persisted_state import load_float, save_float

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

PAIRS = [("ETC", "FIL")]     # only symbol-clean validated pair, see docstring
LOOKBACK_H = 180              # target window; HL testnet ETC 1h history cap fluctuates/shrinks over time
MIN_CANDLES = 100             # adaptive floor -- trade on whatever history is available above this
ENTRY_Z = 2.0
EXIT_Z = 0.5
MAX_HOLD_H = 96
LEVERAGE = 2
PAIR_NOTIONAL_FRAC = 0.55    # raised from 0.35 (2026-08-24): grow capital utilization on the one
                              # validated pair; entry threshold/edge logic (ENTRY_Z) untouched
MIN_NOTIONAL = 11.0
POLL_SECONDS = 3600          # hourly, matches signal timeframe
DD_KILL = 0.15
PEAK_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "logs", ".pairs_arb_peak")
STATE_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "logs", ".pairs_arb_state")
LIVE = os.getenv("PAIRS_ARB_LIVE", "no").lower() == "yes"


class PairsArbExecutor:
    def __init__(self):
        self.addr = os.getenv("HL_WALLET_ADDRESS")
        self.hl = ccxt.hyperliquid({
            "privateKey": os.getenv("HL_PRIVATE") or os.getenv("HL_PRIVATE_KEY"),
            "walletAddress": self.addr,
            "options": {"defaultType": "swap"},
        })
        self.hl.set_sandbox_mode(True)
        self.hl.walletAddress = self.addr
        self.peak_equity = self._load_peak()
        self._last_equity = None   # [RL_METRICS] shadow-only; tracks equity across cycles for reward calc
        self.halted = False
        self.state = self._load_state()   # {"BNB/APT": {"pos": 0, "entry_idx": None}, ...}
        self.db = DatabaseManager()
        self.db.connect()

    def sym(self, b):
        return f"{b}/USDC:USDC"

    def _load_peak(self):
        return load_float(PEAK_FILE, 0.0)

    def _save_peak(self, v):
        save_float(PEAK_FILE, v, "{:.2f}")

    def _load_state(self):
        import json
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except Exception:
            return {f"{a}/{b}": {"pos": 0} for a, b in PAIRS}

    def _save_state(self):
        import json
        with open(STATE_FILE, "w") as f:
            json.dump(self.state, f)

    async def _equity(self):
        b = await self.hl.fetch_balance()
        eq = float(b.get("info", {}).get("marginSummary", {}).get("accountValue", 0) or 0)
        return eq

    async def _closes(self, base, hours):
        candles = await self.hl.fetch_ohlcv(self.sym(base), "1h", limit=hours)
        return np.array([c[4] for c in candles])

    @staticmethod
    def _filled_sz(resp):
        """Actual filled base qty from an HL create_order response (0 on silent no-fill).
        Same pattern as hl_perp_ls.py -- see its docstring for why this can't be trusted blind."""
        info = resp.get("info", {}) if isinstance(resp, dict) else {}
        f = info.get("filled") if isinstance(info, dict) else None
        if isinstance(f, dict):
            try:
                return float(f.get("totalSz") or 0)
            except (TypeError, ValueError):
                return 0.0
        try:
            return float(resp.get("filled") or 0)
        except (TypeError, ValueError):
            return 0.0

    async def _order(self, base, side, qty, reduce_only=False):
        """Verified-fill order: retry unfilled remainder with widening slippage and a refreshed
        price, same guaranteed-fill pattern as hl_perp_ls.py. Returns actual filled base qty."""
        px = float((await self.hl.fetch_ticker(self.sym(base)))["last"])
        log.info(f"[PAIRS_ARB] {'LIVE' if LIVE else 'DRY'} {side} {base} qty={qty:.4f} px={px} reduce_only={reduce_only}")
        if not LIVE:
            return qty
        await self.hl.set_leverage(LEVERAGE, self.sym(base))
        remaining, filled = qty, 0.0
        for attempt, slip in enumerate((0.05, 0.07, 0.09)):
            if remaining * px < 1.0:
                break
            params = {"slippage": slip}
            if reduce_only:
                params["reduceOnly"] = True
            got = 0.0
            try:
                r = await self.hl.create_order(self.sym(base), "market", side, remaining, px, params)
                got = self._filled_sz(r)
            except Exception as e:
                log.error(f"[PAIRS_ARB] {side} {base} try{attempt} slip={slip}: {e}")
            filled += got
            remaining -= got
            log.info(f"[PAIRS_ARB] LIVE {side} {base} try{attempt} slip={slip} "
                     f"filled={got:.6f} cum={filled:.6f}/{qty:.6f} ro={reduce_only}")
            if got <= 0:
                try:
                    px = float((await self.hl.fetch_ticker(self.sym(base)))["last"]) or px
                except Exception:
                    pass
        short = remaining * px
        if short > max(1.0, 0.1 * qty * px):
            log.error(f"[PAIRS_ARB] {side} {base} UNDERFILLED: got {filled:.4f}/{qty:.4f} "
                      f"(short ${short:.2f}) -- will reconcile from exchange truth.")
        return filled

    async def _positions(self, bases):
        """Real signed base qty per symbol, queried fresh from the exchange (truth source) --
        used to reconcile self.state after an exit instead of assuming both legs closed."""
        out = {}
        try:
            for p in await self.hl.fetch_positions([self.sym(b) for b in bases]):
                s = p.get("symbol", "")
                base = s.replace("/USDC:USDC", "")
                amt = float(p.get("contracts", 0) or 0)
                if amt == 0:
                    continue
                side = p.get("side")
                out[base] = amt * (1 if side == "long" else -1)
        except Exception as e:
            log.error(f"[PAIRS_ARB] positions query failed: {e}")
        return out

    async def _reconcile_entry(self, a, b, cur_a, cur_b, side_a, side_b):
        """After entry orders, verify both legs landed a comparable notional. A margin-blocked
        underfill on one leg (like the FIL insufficient-margin case) otherwise leaves a naked
        directional position instead of a hedged spread. Top up the short leg; if that also
        fails, unwind the other leg down to match so the pair stays market-neutral."""
        real = await self._positions([a, b])
        notional = {a: abs(real.get(a, 0.0)) * cur_a, b: abs(real.get(b, 0.0)) * cur_b}
        small, big = (a, b) if notional[a] < notional[b] else (b, a)
        diff = notional[big] - notional[small]
        if diff <= max(MIN_NOTIONAL, 0.1 * max(notional[big], 1.0)):
            return
        small_cur, small_side = (cur_a, side_a) if small == a else (cur_b, side_b)
        topup_qty = diff / small_cur if small_cur > 0 else 0
        got = await self._order(small, small_side, topup_qty)
        if got * small_cur >= diff * 0.5:
            return
        real2 = await self._positions([a, b])
        notional2 = {a: abs(real2.get(a, 0.0)) * cur_a, b: abs(real2.get(b, 0.0)) * cur_b}
        excess = notional2[big] - notional2[small]
        if excess > MIN_NOTIONAL:
            big_cur = cur_a if big == a else cur_b
            entry_side = side_a if big == a else side_b
            unwind_side = "buy" if entry_side == "sell" else "sell"
            unwind_qty = excess / big_cur
            await self._order(big, unwind_side, unwind_qty, reduce_only=True)
            log.error(f"[PAIRS_ARB] {a}/{b} entry imbalance: unwound ${excess:.2f} on {big} leg to restore neutrality.")

    def _log_open(self, base, side, px, qty):
        if not LIVE:
            return
        try:
            self.db.insert_trade(f"{base}/USDT", "hyperliquid", side, float(px), float(qty), "LIVE_OK",
                                  metadata={"quant_signals": {"strategy": "pairs_arb"}, "reduce_only": False})
        except Exception as e:
            log.error(f"[PAIRS_ARB] DB open-log failed for {base}: {e}")

    def _log_close(self, base, roi):
        if not LIVE:
            return
        try:
            self.db.log_trade_outcome(f"{base}/USDT", "hyperliquid", float(roi))
        except Exception as e:
            log.error(f"[PAIRS_ARB] DB outcome-log failed for {base}: {e}")

    async def check_pair(self, a, b, equity):
        key = f"{a}/{b}"
        closes_a = await self._closes(a, LOOKBACK_H + 1)
        closes_b = await self._closes(b, LOOKBACK_H + 1)
        if len(closes_a) < MIN_CANDLES or len(closes_b) < MIN_CANDLES:
            log.warning(f"[PAIRS_ARB] {key} insufficient candles (a={len(closes_a)} b={len(closes_b)}), skipping.")
            return
        n = min(len(closes_a), len(closes_b))
        closes_a, closes_b = closes_a[-n:], closes_b[-n:]

        window_a, window_b = closes_a[:-1], closes_b[:-1]
        hedge = float(np.polyfit(window_b, window_a, 1)[0])
        spread = window_a - hedge * window_b
        mu, sigma = spread.mean(), spread.std()
        if sigma == 0:
            return

        cur_a, cur_b = closes_a[-1], closes_b[-1]
        z = ((cur_a - hedge * cur_b) - mu) / sigma

        st = self.state.get(key, {"pos": 0})
        pos = st.get("pos", 0)
        # Exit decisions on an already-open position must judge reversion against the SAME
        # spread definition it was entered under, not a freshly re-estimated hedge/mu/sigma
        # (which drifts cycle to cycle and can mask real directional loss as "not reverted yet").
        # Entry decisions below still use the fresh rolling z (correct: no position exists yet).
        if pos != 0 and "mu" in st and "sigma" in st:
            entry_hedge, entry_mu, entry_sigma = st["hedge"], st["mu"], st["sigma"]
            z_exit = ((cur_a - entry_hedge * cur_b) - entry_mu) / entry_sigma if entry_sigma else z
        else:
            z_exit = z
        notional = equity * PAIR_NOTIONAL_FRAC * LEVERAGE
        qty_a = notional / cur_a if cur_a > 0 else 0
        # Hedge-matched quantity: spread = price_a - hedge*price_b, so qty_b = qty_a*hedge keeps the
        # position tracking the spread (not dollar-matched to leg B's own price). Fixed 2026-08-25:
        # old formula (notional*hedge/cur_b) was off by a factor of cur_a/cur_b (~11x on ETC/FIL),
        # causing an oversized FIL order that blew through available margin.
        qty_b = qty_a * abs(hedge)

        z_exit_suffix = f" z_exit={z_exit:.2f}" if pos != 0 else ""
        log.info(f"[PAIRS_ARB] {key} z={z:.2f} hedge={hedge:.3f} pos={pos} notional=${notional:.2f}{z_exit_suffix}")

        # --- [RL_METRICS] shadow logging (additive-only; reuses cycle-level equity + this pair's z,
        # no new API calls). Sortino/drawdown-penalty style reward-shaping, in the spirit of
        # src/quant/reward_scheme.py, computed inline for observability only. ---
        try:
            reward = (equity - self._last_equity) if self._last_equity is not None else 0.0
            z_penalty = max(0.0, abs(z) - ENTRY_Z)                      # risk-exposure beyond entry threshold
            dd_penalty = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0.0
            penalty = z_penalty + dd_penalty
            net_score = reward - penalty
            log.info(f"[RL_METRICS] agent=pairs_arb pair={key} cycle_ts={time.time():.0f} "
                     f"reward=${reward:+.2f} penalty={penalty:+.3f} net_score={net_score:+.2f} z={z:+.2f}")
        except Exception as e:
            log.warning(f"[RL_METRICS] agent=pairs_arb pair={key} metric computation failed: {e}")

        if pos == 0:
            if notional < MIN_NOTIONAL:
                return
            if z > ENTRY_Z:
                # spread too high -> short a, long b
                await self._order(a, "sell", qty_a)
                await self._order(b, "buy", qty_b)
                await self._reconcile_entry(a, b, cur_a, cur_b, "sell", "buy")
                self.state[key] = {"pos": -1, "entry_idx": 0, "hedge": hedge, "mu": mu, "sigma": sigma, "entry_a": cur_a, "entry_b": cur_b}
                self._log_open(a, "sell", cur_a, qty_a)
                self._log_open(b, "buy", cur_b, qty_b)
            elif z < -ENTRY_Z:
                await self._order(a, "buy", qty_a)
                await self._order(b, "sell", qty_b)
                await self._reconcile_entry(a, b, cur_a, cur_b, "buy", "sell")
                self.state[key] = {"pos": 1, "entry_idx": 0, "hedge": hedge, "mu": mu, "sigma": sigma, "entry_a": cur_a, "entry_b": cur_b}
                self._log_open(a, "buy", cur_a, qty_a)
                self._log_open(b, "sell", cur_b, qty_b)
        else:
            held = st.get("entry_idx", 0) + 1
            self.state[key]["entry_idx"] = held
            mean_reverted = (pos == 1 and z_exit >= -EXIT_Z) or (pos == -1 and z_exit <= EXIT_Z)
            stale = held >= MAX_HOLD_H
            if mean_reverted or stale:
                # Each leg gets its own try/except -- a raise on leg B must not skip the
                # exchange-truth reconciliation below (that gap is what caused the state desync).
                try:
                    await self._order(a, "buy" if pos == -1 else "sell", qty_a, reduce_only=True)
                except Exception as e:
                    log.error(f"[PAIRS_ARB] exit leg {a} failed: {e}")
                try:
                    await self._order(b, "sell" if pos == -1 else "buy", qty_b, reduce_only=True)
                except Exception as e:
                    log.error(f"[PAIRS_ARB] exit leg {b} failed: {e}")

                # Feedback loop: log ROI for both legs against their stored entry price.
                entry_a, entry_b = st.get("entry_a"), st.get("entry_b")
                if entry_a and entry_b:
                    roi_a = pos * (cur_a - entry_a) / entry_a
                    roi_b = -pos * (cur_b - entry_b) / entry_b
                    self._log_close(a, roi_a)
                    self._log_close(b, roi_b)

                # Reconcile against real exchange state instead of assuming both legs closed.
                real = await self._positions([a, b])
                flat_a = abs(real.get(a, 0.0) * cur_a) < MIN_NOTIONAL
                flat_b = abs(real.get(b, 0.0) * cur_b) < MIN_NOTIONAL
                if flat_a and flat_b:
                    self.state[key] = {"pos": 0}
                else:
                    log.error(f"[PAIRS_ARB] {key} exit INCOMPLETE: real_{a}={real.get(a, 0.0):.4f} "
                              f"real_{b}={real.get(b, 0.0):.4f} -- state kept open, retrying next cycle.")

        self._save_state()

    async def cycle(self):
        equity = await self._equity()
        if equity > self.peak_equity:
            self.peak_equity = equity
            self._save_peak(self.peak_equity)
        if self.peak_equity > 0 and equity <= self.peak_equity * (1 - DD_KILL):
            log.error(f"[PAIRS_ARB] DD KILL: equity ${equity:.2f} <= {(1-DD_KILL)*100:.0f}% of peak ${self.peak_equity:.2f}. Halting (manual flatten required).")
            self.halted = True
            return
        for a, b in PAIRS:
            try:
                await self.check_pair(a, b, equity)
            except Exception as e:
                log.error(f"[PAIRS_ARB] {a}/{b} cycle error: {e}")
        self._last_equity = equity   # [RL_METRICS] shadow-only; advance once per cycle after all pairs checked

    async def run_forever(self):
        log.info(f"Starting Pairs Stat-Arb (LIVE={LIVE}) pairs={PAIRS} entry_z={ENTRY_Z} exit_z={EXIT_Z}")
        try:
            while True:
                if os.path.exists("state/STALE_WAKE_HALT") and not self.halted:
                    log.error("[PAIRS_ARB] STALE_WAKE_HALT marker present (watchdog was silent -- host likely slept). Halting until cleared.")
                    self.halted = True
                elif not os.path.exists("state/STALE_WAKE_HALT") and self.halted:
                    log.info("[PAIRS_ARB] STALE_WAKE_HALT cleared — resuming trading.")
                    self.halted = False
                if self.halted:
                    log.error("[PAIRS_ARB] halted by kill-switch; sleeping without trading.")
                else:
                    await self.cycle()
                await asyncio.sleep(POLL_SECONDS)
        finally:
            await self.hl.close()

    async def dry_scan(self):
        await self.cycle()
        await self.hl.close()


if __name__ == "__main__":
    asyncio.run(PairsArbExecutor().run_forever() if LIVE else PairsArbExecutor().dry_scan())
