"""
Hyperliquid TESTNET dollar-neutral cross-sectional PERP long-short executor.

Context: HL testnet perp PRICES are real (oracle-fed); funding + spot are fake. So on testnet
the tradeable, meaningful signal is PRICE-based. Delta-neutral carry is impossible here (needs
real spot), so this trades perps only.

Design maps to the staged plan:
  1. CONTROL BLEEDING  -> dollar-neutral by construction (sum long notional == sum short notional):
     market beta ~= 0, so a market crash/pump cannot wipe the book. Plus a hard drawdown
     kill-switch that flattens everything and halts.
  2. CONTROL DUST      -> every leg must clear MIN_NOTIONAL; rebalance deltas below REBAL_MIN_USD
     are skipped so the book does not churn fees on noise.
  3. STEADY GROWTH     -> cross-sectional momentum tilt (long strongest, short weakest) is the
     return source. Signal is pluggable; edge is validated on the parallel research track.
  4. PARALLEL DATA     -> every rebalance + fill is logged for the backtest/training loop.

Signal: cross-sectional momentum over LOOKBACK_H hours of real oracle prices. Rank the universe,
long the top QUANTILE, short the bottom QUANTILE, equal notional per name, dollar-neutral.

SAFETY: testnet only (set_sandbox_mode(True)). Leverage low (default 2x). Exits/trims are
reduce_only. Dry-run by default; set HL_PERP_LIVE=yes to send orders.

Run (dry):  venv/bin/python -m src.execution.hl_perp_ls
Run (live): HL_PERP_LIVE=yes venv/bin/python -m src.execution.hl_perp_ls
"""
import asyncio
import os
import time
import numpy as np
import ccxt.async_support as ccxt
from dotenv import load_dotenv
from src.utils.logger import log
from src.utils.db import DatabaseManager
from src.utils.persisted_state import load_float, save_float, load_json, save_json

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

# CARRY_RESERVED: the carry executor (src/execution/carry_executor.py) trades BTC/ETH/SOL perp
# shorts out of the SAME perp wallet as this book. If those symbols were in this UNIVERSE, both
# strategies would open/trim/reduce_only the same net position and fight each other (e.g. carry's
# SOL short would look like this book's SOL leg and get trimmed by the neutrality-repair). So we
# carve them out: carry OWNS majors, this book trades only the alts. Zero shared-symbol coupling.
# (Margin is still shared, but GROSS_FRACTION=0.90 already reserves a ~10% buffer > carry's tiny
# ~$33 1x notional, so the sizing overlap is immaterial.)
CARRY_RESERVED = ["BTC", "ETH", "SOL"]
# NEAR removed: testnet book too thin for shorts — all short attempts fail with
# "Price too far from oracle" / "no resting orders", leaving the book unhedged.
UNIVERSE = ["BNB", "DOGE", "AVAX", "ARB", "OP", "ADA", "APT", "ATOM", "INJ"]
LOOKBACK_H = 72                 # momentum formation window (hours)
QUANTILE = 0.30                 # long top 30% / short bottom 30%
HYSTERESIS = 0.15               # incumbent name stays in its basket until it falls out of the
                                # top/bottom (QUANTILE+HYSTERESIS) band. Kills churn from marginal
                                # names rotating in/out every cycle (the dominant cost bleed).
LEVERAGE = 1.5                  # raised from 1 (2026-08-24): quantile-basket/dust-lockout fix now lets
                                # gross actually deploy, so revisit growth vs the 2026-08-20 funding-cost
                                # cut. Partial restore (not full 2x/0.90) since the original signal-too-thin
                                # concern re: funding drag still applies -- DD_KILL remains the hard stop.
GROSS_FRACTION = 0.65           # raised from 0.45 (2026-08-24), same reasoning as LEVERAGE above
MIN_NOTIONAL = 11.0             # HL min ~$10; buffer
REBAL_MIN_USD = 15.0            # skip position deltas smaller than this (dust/churn guard).
                                # Raised from 6.0 (2026-08-19): ATTRIB logs showed cost(funding+fees)
                                # negative on 16/20 cycles avg -$1.10, while signal_pnl was only
                                # +$1.56 net over 5 days on ~$220 gross -- taker churn on sub-$6 noise
                                # deltas (~20% of a typical $30 leg) was eating the whole edge.
REBALANCE_SECONDS = 4 * 3600    # rebalance every 4h. Market orders are taker: churning the whole
                                # book every 30 min bled ~0.3%/cycle of gross in slippage+fees and
                                # was the entire (negative) P&L. Momentum is a multi-hour signal;
                                # 4h captures it while cutting transaction cost ~8x.
DD_KILL = 0.15                  # flatten + halt if equity drops 15% from peak
# Persistent high-water mark: peak_equity MUST survive process restarts, else every watchdog
# respawn resets the reference and the drawdown kill only ever catches 15% from the latest
# restart -- never the cumulative bleed. This file is the cross-restart memory.
PEAK_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "logs", ".perp_ls_peak")
# One-time marker: after the CARRY_RESERVED carve-out, this book must close any leftover major
# positions it still holds ONCE, then never again -- so a later restart can't flatten a legit
# carry position. Presence of this file = cleanup already done.
RECONCILE_MARKER = os.path.join(os.path.dirname(__file__), "..", "..", "logs", ".perp_ls_reserved_reconciled")
# Cumulative gross-momentum (price-only) P&L, persisted across restarts. This is the SIGNAL edge
# stripped of funding/fees: if it trends positive while net equity bleeds, the momentum signal
# works and only the (fake, on testnet) funding is killing net P&L -- the honest edge proof.
SIGNAL_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "logs", ".perp_ls_cum_signal")

# --- edge-vs-cost adaptive overlay (2026-08-28): net_score logged consistently negative
# (funding+fees structurally outweigh the price-timing edge every cycle) plus a recurring
# neutrality-repair skew bug (flat $REBAL_MIN_USD trigger fires on tiny skew relative to gross,
# paying double taker slippage). This block adds a rolling edge/cost ratio that throttles gross,
# a minimum-edge trade gate, adaptive cadence + a %-of-gross repair tolerance band, and a
# rolling promotion gate -- all additive to the existing vol/dd risk overlay above.
EDGE_COST_WINDOW = 20            # rolling cycles used for the edge/cost ratio (NOT since-inception)
EDGE_COST_MIN_SAMPLES = 3        # need at least this many cycles of history before gating on it
EDGE_COST_FLOOR = 1.0            # ratio at/below this -> edge_scale floors out (edge doesn't cover cost)
EDGE_COST_CEIL = 1.3             # ratio at/above this -> edge_scale caps at 1.0 (edge comfortably covers cost)
EDGE_SCALE_FLOOR = 0.15          # never fully zero gross out on a weak ratio; leaves room to recover
MIN_EDGE_K = 1.75                # skip a leg's entry/add if its expected $ edge < k * trailing avg cost/trade
SKEW_REPAIR_PCT = 5.0            # tolerance band: don't force-repair skew under this % of gross
# Repair-path dollar floor, deliberately smaller than REBAL_MIN_USD (dust/churn guard). The
# original bug: repair reused REBAL_MIN_USD both as the gate to even attempt a repair AND as the
# per-cut floor inside it, so a $12 net skew over $15 never repaired even after clearing the
# SKEW_REPAIR_PCT band -- REBAL_MIN_USD was sized for entry/exit noise, not for a repair trim that
# is explicitly risk-reducing and already past the %-of-gross tolerance check above it.
# Must still clear HL's own exchange-side minimum order value ($10) -- a first live repair attempt
# at 5.0 tried to cut $6.05 and got rejected ("Order must have minimum value of $10"). Match
# MIN_NOTIONAL (11.0, the $10 min + buffer already used elsewhere in this file) instead.
REPAIR_MIN_USD = MIN_NOTIONAL
CADENCE_MIN_SECONDS = 2 * 3600   # adaptive-cadence floor (compress no faster than this)
CADENCE_MAX_SECONDS = 8 * 3600   # adaptive-cadence ceiling (stretch no slower than this)
PROMOTION_WINDOW = 20            # rolling cycles for the promotion-gate averages
PROMOTION_CONSECUTIVE = 3        # consecutive passing windows required before PROMOTION_READY
EDGE_COST_HISTORY_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "logs", ".perp_ls_edge_cost_hist")
PROMOTION_STREAK_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "logs", ".perp_ls_promotion_streak")
CYCLE_COUNT_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "logs", ".perp_ls_cycle_count")

# --- edge_scale floor probation (chicken-and-egg fix): once edge_cost_ratio dips to/below
# EDGE_COST_FLOOR, edge_scale pins at EDGE_SCALE_FLOOR forever -- the book stays too small for
# signal P&L to ever outrun fixed cost, so the ratio can never recover on its own. Periodically
# force a larger trial size regardless of the ratio so the strategy gets an honest chance to prove
# (or disprove) itself at meaningful size instead of being starved indefinitely.
PROBATION_EVERY_N = 6            # roughly once/day at the ~4-6h base cadence
PROBATION_SCALE = 0.5            # trial edge_scale for a probation cycle (vs EDGE_SCALE_FLOOR=0.15)

LIVE = os.getenv("HL_PERP_LIVE", "no").lower() == "yes"


class PerpLongShort:
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
        self.halted = False
        # P&L attribution state (see SIGNAL_FILE). Book is static between 4h rebalances, so the
        # last snapshot's signed notional * price-return = exact price-only interval P&L.
        self.last_notional = {}     # {base: signed_usd_notional} held over the last interval
        self.last_px = {}           # {base: price at last snapshot}
        self.last_equity = 0.0
        self.cum_signal = self._load_cum_signal()
        # rolling edge/cost history: list of {signal, cost, net_score, legs} dicts, newest last,
        # trimmed to EDGE_COST_WINDOW / PROMOTION_WINDOW (kept as one list sized to the larger).
        self.edge_cost_history = self._load_edge_cost_history()
        self.promotion_streak = int(load_float(PROMOTION_STREAK_FILE, 0.0))
        self.cycle_count = int(load_float(CYCLE_COUNT_FILE, 0.0))
        self.next_cadence = REBALANCE_SECONDS
        self.last_legs_traded = 0   # legs traded in the previous cycle's main order loop
        self.db = DatabaseManager()
        self.db.connect()

    def sym(self, b):
        return f"{b}/USDC:USDC"

    def _load_peak(self):
        """Restore the cross-restart high-water mark. Absent file -> 0.0, which the first
        rebalance seeds to current equity (fresh baseline for this deploy)."""
        return load_float(PEAK_FILE, 0.0)

    def _save_peak(self, v):
        save_float(PEAK_FILE, v, "{:.2f}")

    def _load_cum_signal(self):
        return load_float(SIGNAL_FILE, 0.0)

    def _save_cum_signal(self, v):
        save_float(SIGNAL_FILE, v, "{:.4f}")

    def _load_edge_cost_history(self):
        h = load_json(EDGE_COST_HISTORY_FILE, [])
        return h if isinstance(h, list) else []

    def _save_edge_cost_history(self):
        keep = max(EDGE_COST_WINDOW, PROMOTION_WINDOW)
        self.edge_cost_history = self.edge_cost_history[-keep:]
        save_json(EDGE_COST_HISTORY_FILE, self.edge_cost_history)

    def _rolling_edge_cost(self, window):
        """Sum signal/cost over the trailing `window` cycles (NOT since-inception). Returns
        (ratio, rolling_signal, rolling_cost, avg_cost_per_trade, n_samples)."""
        recent = self.edge_cost_history[-window:]
        n = len(recent)
        rolling_signal = sum(e.get("signal", 0.0) for e in recent)
        rolling_cost = sum(abs(e.get("cost", 0.0)) for e in recent)
        legs = sum(e.get("legs", 0) for e in recent)
        eps = 1e-6
        ratio = rolling_signal / max(rolling_cost, eps)
        avg_cost_per_trade = rolling_cost / legs if legs > 0 else 0.0
        return ratio, rolling_signal, rolling_cost, avg_cost_per_trade, n

    def _edge_scale(self, ratio, n):
        """Shrinks gross toward EDGE_SCALE_FLOOR when the rolling edge/cost ratio is weak,
        grows toward 1.0 (capped) once edge comfortably covers cost. Neutral (1.0) until enough
        rolling history exists to trust the ratio."""
        if n < EDGE_COST_MIN_SAMPLES:
            return 1.0
        if ratio <= EDGE_COST_FLOOR:
            return EDGE_SCALE_FLOOR
        if ratio >= EDGE_COST_CEIL:
            return 1.0
        frac = (ratio - EDGE_COST_FLOOR) / (EDGE_COST_CEIL - EDGE_COST_FLOOR)
        return EDGE_SCALE_FLOOR + frac * (1.0 - EDGE_SCALE_FLOOR)

    def _adaptive_cadence(self, vol, target_vol, ratio, n):
        """Stretch rebalance cadence when vol is low AND edge is weak (nothing worth chasing,
        save cost); compress only when both vol and edge justify trading more often. Neutral
        (base REBALANCE_SECONDS) until enough rolling history exists."""
        if n < EDGE_COST_MIN_SAMPLES:
            return REBALANCE_SECONDS
        factor = 1.0
        if ratio < EDGE_COST_FLOOR:
            factor *= 1.5
        elif ratio > EDGE_COST_CEIL:
            factor *= 0.75
        if vol < target_vol * 0.5:
            factor *= 1.3
        elif vol > target_vol * 1.5:
            factor *= 0.8
        cadence = REBALANCE_SECONDS * factor
        return min(CADENCE_MAX_SECONDS, max(CADENCE_MIN_SECONDS, cadence))

    async def _equity(self):
        b = await self.hl.fetch_balance()
        return float(b.get("total", {}).get("USDC", 0) or 0), float(b.get("free", {}).get("USDC", 0) or 0)

    async def _market_metrics(self):
        """
        Computes cross-sectional momentum signals, trailing realized hourly volatility,
        and maximum trailing drawdown of the equal-weight universe price index.
        """
        closes_by_sym = {}
        for b in UNIVERSE:
            try:
                oh = await self.hl.fetch_ohlcv(self.sym(b), "1h", limit=LOOKBACK_H + 2)
                closes = [c[4] for c in oh]
                if len(closes) >= LOOKBACK_H:
                    closes_by_sym[b] = closes[-LOOKBACK_H:]
            except Exception as e:
                log.error(f"[PERP_LS] fetch ohlcv {b}: {e}")
                
        if not closes_by_sym:
            return {}, 0.0, 0.0
            
        # Calculate momentum signals
        mom = {}
        for b, closes in closes_by_sym.items():
            mom[b] = closes[-1] / closes[0] - 1.0
            
        # Compute equal-weight hourly returns and index path
        n_steps = min(len(c) for c in closes_by_sym.values())
        hourly_returns = []
        index_value = 1.0
        index_path = [1.0]
        
        for t in range(1, n_steps):
            step_returns = []
            for b in closes_by_sym:
                p_prev = closes_by_sym[b][t - 1]
                p_now = closes_by_sym[b][t]
                if p_prev > 0:
                    step_returns.append((p_now - p_prev) / p_prev)
            if step_returns:
                ret = float(np.mean(step_returns))
                hourly_returns.append(ret)
                index_value *= (1 + ret)
                index_path.append(index_value)
                
        # Trailing realized volatility (hourly)
        vol = float(np.std(hourly_returns)) if hourly_returns else 0.0
        
        # Trailing index drawdown
        index_arr = np.array(index_path)
        peak = np.maximum.accumulate(index_arr)
        dd = (index_arr - peak) / peak
        max_dd = abs(float(dd.min())) if len(dd) > 0 else 0.0
        
        return mom, vol, max_dd

    async def _positions(self):
        """Current signed notional per base: {base: usd_notional (+long/-short)}."""
        out = {}
        try:
            for p in await self.hl.fetch_positions():
                s = p.get("symbol", "")
                base = s.replace("/USDC:USDC", "")
                if base not in UNIVERSE:
                    continue
                amt = float(p.get("contracts", 0) or 0)
                if amt == 0:
                    continue
                side = p.get("side")
                px = float(p.get("markPrice") or p.get("entryPrice") or 0)
                notion = amt * px * (1 if side == "long" else -1)
                out[base] = notion
        except Exception as e:
            log.error(f"[PERP_LS] positions: {e}")
        return out

    def _target_notionals(self, mom, gross, cur=None):
        """Dollar-neutral target: +gross/2 spread across longs, -gross/2 across shorts.

        Hysteresis: a name currently HELD stays in its basket until it falls out of the wider
        (QUANTILE+HYSTERESIS) rank band, instead of being dropped the moment it leaves the strict
        top/bottom QUANTILE. Without this, marginal names flip in/out every cycle and each flip is
        a full taker exit + full taker entry -- the churn that was the dominant cost bleed."""
        cur = cur or {}
        ranked = sorted(mom.items(), key=lambda kv: kv[1])   # ascending: weakest momentum first
        order = [b for b, _ in ranked]
        n = len(order)
        k = max(1, int(n * QUANTILE))
        # Budget-aware basket cap (2026-08-24): when the risk overlay throttles gross low, splitting
        # it across the full QUANTILE basket can put each leg under MIN_NOTIONAL, so the dust filter
        # zeroed every leg out (longs=[] lockout) even though the momentum signal was fine. Shrink the
        # basket so per-leg notional clears MIN_NOTIONAL instead of silently dropping the whole side.
        if gross > 0:
            k_afford = max(1, int((gross / 2) // MIN_NOTIONAL))
            k = min(k, k_afford)
        kb = max(k, int(n * (QUANTILE + HYSTERESIS)))         # wider retention band for incumbents
        if gross > 0:
            kb = max(k, min(kb, k_afford))
        short_core, short_band = set(order[:k]), set(order[:kb])
        long_core, long_band = set(order[-k:]), set(order[-kb:])

        longs, shorts = [], []
        for b in mom:
            held = cur.get(b, 0.0)
            if b in long_core or (held > 0 and b in long_band):
                longs.append(b)
            elif b in short_core or (held < 0 and b in short_band):
                shorts.append(b)
        if not longs:                                         # safety: never empty a side
            longs = order[-k:]
        if not shorts:
            shorts = order[:k]

        per_long = (gross / 2) / len(longs)
        per_short = (gross / 2) / len(shorts)
        tgt = {b: 0.0 for b in mom}
        for b in longs:
            tgt[b] = per_long
        for b in shorts:
            tgt[b] = -per_short
        # skip legs below min notional (dust control)
        return {b: (v if abs(v) >= MIN_NOTIONAL else 0.0) for b, v in tgt.items()}

    @staticmethod
    def _filled_sz(resp):
        """Actual filled size from an HL create_order response. HL raw status is one of
        {"filled":{"totalSz",...}} | {"resting":{...}} | {"error":"..."}. A market IOC that
        finds no depth within the slippage band returns NO filled key -> 0 (this is the silent
        no-fill that broke dollar-neutrality). Returns filled base qty (float)."""
        info = resp.get("info", {}) if isinstance(resp, dict) else {}
        f = info.get("filled") if isinstance(info, dict) else None
        if isinstance(f, dict):
            try:
                return float(f.get("totalSz") or 0)
            except (TypeError, ValueError):
                return 0.0
        try:                                   # ccxt-normalized fallback
            return float(resp.get("filled") or 0)
        except (TypeError, ValueError):
            return 0.0

    async def _order(self, base, delta_usd, px, reduce_only=False):
        """Send a market order and GUARANTEE the fill (or report the shortfall). Testnet perp
        books are thin/asymmetric: a single IOC at px*(1+slip) can silently no-fill. So we send,
        read the real filled size, and retry the unfilled remainder with a wider slippage band
        and a refreshed price until it clears or we run out of attempts. Returns filled USD."""
        target_qty = abs(delta_usd) / px
        side = "buy" if delta_usd > 0 else "sell"
        if not LIVE:
            log.info(f"[PERP_LS] DRY {side} {base} ${abs(delta_usd):.2f} qty={target_qty:.6f} px={px} ro={reduce_only}")
            return abs(delta_usd)

        # The caller's px is a snapshot from the top of the rebalance cycle; by the time this leg's
        # order fires (after prior legs' awaits) it can be stale enough to already sit outside HL's
        # oracle band on a thin/volatile testnet name, causing the first attempt to fail before any
        # retry even widens slippage. One fresh fetch here costs a single round-trip and removes that
        # avoidable first-try rejection (recurring "Price too far from oracle" on NEAR was this).
        try:
            px = float((await self.hl.fetch_ticker(self.sym(base)))["last"]) or px
            target_qty = abs(delta_usd) / px
        except Exception as e:
            log.warning(f"[PERP_LS] pre-order price refresh failed for {base}: {e}")

        # Entry price BEFORE this fill, needed to compute ROI once the close/trim actually lands.
        entry_px = None
        if reduce_only:
            try:
                for p in await self.hl.fetch_positions([self.sym(base)]):
                    if p.get("symbol", "").replace("/USDC:USDC", "") == base:
                        entry_px = float(p.get("entryPrice") or 0) or None
                        break
            except Exception as e:
                log.error(f"[PERP_LS] entry price lookup failed for {base}: {e}")

        remaining = target_qty
        filled = 0.0
        # slippage MUST stay inside HL's oracle band -- a marketable limit too far from oracle is
        # rejected outright ("Price too far from oracle"). So retries widen only slightly; extra
        # tries mainly give the thin testnet book time to refresh at a fresh price.
        for attempt, slip in enumerate((0.04, 0.06, 0.08)):
            if remaining * px < 1.0:            # negligible dust left
                break
            params = {"slippage": slip}
            if reduce_only:
                params["reduceOnly"] = True
            got = 0.0
            try:
                r = await self.hl.create_order(self.sym(base), "market", side, remaining, px, params)
                got = self._filled_sz(r)
            except Exception as e:
                log.error(f"[PERP_LS] {side} {base} try{attempt} slip={slip}: {e}")
            filled += got
            remaining -= got
            log.info(f"[PERP_LS] LIVE {side} {base} try{attempt} slip={slip} "
                     f"filled={got:.6f} cum={filled:.6f}/{target_qty:.6f} ro={reduce_only}")
            if got <= 0:                        # book didn't cross; refresh price and widen
                try:
                    px = float((await self.hl.fetch_ticker(self.sym(base)))["last"]) or px
                except Exception:
                    pass
        short_usd = remaining * px
        if short_usd > max(1.0, 0.1 * abs(delta_usd)):
            log.error(f"[PERP_LS] {side} {base} UNDERFILLED: got ${filled*px:.2f} of ${abs(delta_usd):.2f} "
                      f"(short ${short_usd:.2f}) -- leg not fully hedged; will reconcile next rebalance.")

        # Feedback loop: log every real fill so the meta-learner / LLM critique / audit have data
        # from this book. Opens/adds get a fresh row; reduce_only closes get an ROI outcome logged
        # against the most recent open row (falls back to a bare LIVE_OK row if no entry price).
        if filled > 0:
            market_id = f"{base}/USDT"
            try:
                meta = {"reduce_only": reduce_only}
                if not reduce_only:
                    meta["quant_signals"] = {"strategy": "perp_ls"}  # marks this row as an "open" for outcome-matching
                self.db.insert_trade(market_id, "hyperliquid", side, px, filled, "LIVE_OK", metadata=meta)
                if reduce_only and entry_px:
                    # side=="buy" reduce_only closes a short (profit if fill < entry); "sell" closes a long.
                    roi = (entry_px - px) / entry_px if side == "buy" else (px - entry_px) / entry_px
                    self.db.log_trade_outcome(market_id, "hyperliquid", roi)
            except Exception as e:
                log.error(f"[PERP_LS] DB trade logging failed for {base}: {e}")

        return filled * px

    async def rebalance(self):
        equity, free = await self._equity()
        if equity > self.peak_equity:
            self.peak_equity = equity
            self._save_peak(self.peak_equity)   # persist new high-water across restarts
        # --- bleeding kill-switch ---
        if self.peak_equity > 0 and equity <= self.peak_equity * (1 - DD_KILL):
            log.error(f"[PERP_LS] DD KILL: equity ${equity:.2f} <= {(1-DD_KILL)*100:.0f}% of peak ${self.peak_equity:.2f}. Flattening + halt.")
            await self.flatten()
            self.halted = True
            return

        mom, vol, max_dd = await self._market_metrics()
        if len(mom) < 4:
            log.warning(f"[PERP_LS] only {len(mom)} signals; skip rebalance.")
            return

        self.cycle_count += 1
        save_float(CYCLE_COUNT_FILE, float(self.cycle_count), "{:.0f}")


        # Volatility target (hourly std dev target = 0.008, equivalent to ~0.8% hourly / 8% daily vol)
        target_vol = 0.008
        m_vol = 1.0 if vol <= 0 else min(1.0, target_vol / vol)
        
        # Smooth Drawdown Scaling (scale down to a floor of 35% between 8% and 22% drawdown)
        # eased 2026-08-24: market drawdown recovering, prior 5%/15%/20% band was choking gross
        # so hard it fell below MIN_NOTIONAL per-leg after quantile split (longs=[] lockout).
        dd_start = 0.08
        dd_max = 0.22
        expo_floor = 0.35
        
        if max_dd <= dd_start:
            m_dd = 1.0
        elif max_dd >= dd_max:
            m_dd = expo_floor
        else:
            m_dd = 1.0 - ((max_dd - dd_start) / (dd_max - dd_start)) * (1.0 - expo_floor)
            
        # rolling edge/cost ratio -> edge_scale (uses history from PRIOR cycles only; this
        # cycle's own signal/cost gets appended to history further below, after ATTRIB).
        ec_ratio, ec_signal, ec_cost, avg_cost_per_trade, ec_n = self._rolling_edge_cost(EDGE_COST_WINDOW)
        m_edge = self._edge_scale(ec_ratio, ec_n)

        # probation: a floored ratio can never recover on its own (book too small to prove edge ->
        # ratio stays low -> book stays small). Periodically override the floor for one cycle so the
        # strategy gets an honest shot at meaningful size instead of being starved indefinitely.
        is_probation = (ec_n >= EDGE_COST_MIN_SAMPLES and ec_ratio <= EDGE_COST_FLOOR
                         and self.cycle_count % PROBATION_EVERY_N == 0)
        if is_probation:
            log.info(f"[PERP_LS] PROBATION cycle ({self.cycle_count}): overriding edge_scale "
                     f"{m_edge:.2f}x -> {PROBATION_SCALE:.2f}x to test larger size.")
            m_edge = PROBATION_SCALE

        m = m_vol * m_dd * m_edge
        gross = equity * LEVERAGE * GROSS_FRACTION * m
        log.info(f"[PERP_LS] Risk Overlay Metrics: trailing_vol={vol*100:.3f}% (target: {target_vol*100:.2f}%), trailing_drawdown={max_dd*100:.2f}%, edge_cost_ratio={ec_ratio:.2f} (n={ec_n}). Multipliers: vol_scale={m_vol:.2f}x, drawdown_scale={m_dd:.2f}x, edge_scale={m_edge:.2f}x -> composite_scale={m:.2f}x (gross_target: ${gross:.2f})")

        self.next_cadence = self._adaptive_cadence(vol, target_vol, ec_ratio, ec_n)
        
        cur = await self._positions()
        tgt = self._target_notionals(mom, gross, cur)     # hysteresis needs current holdings
        # Fetch tickers for mom UNION cur, not just mom: a base can be currently HELD but missing
        # from mom this cycle (its OHLCV fetch failed/short in _market_metrics), and if so it must
        # still be visible to the resize loop below or the position is orphaned -- silently stuck
        # at its old size forever instead of shrinking/closing toward the fresh gross_target.
        book_bases = set(mom) | set(cur)
        tickers = await self.hl.fetch_tickers([self.sym(b) for b in book_bases])

        # --- P&L attribution: price-only momentum P&L vs realized net (isolates funding/fee cost) ---
        # Book is static between rebalances, so signed_notional * price-return = exact interval
        # signal P&L. cost = net equity change - signal P&L = funding + slippage (+ tiny carry-wallet
        # noise, shared perp wallet). cum_signal>0 while net bleeds => signal works, funding kills it.
        if self.last_notional and self.last_equity > 0:
            signal_pnl = 0.0
            for b, n0 in self.last_notional.items():
                p0 = self.last_px.get(b, 0.0)
                p1 = float((tickers.get(self.sym(b)) or {}).get("last") or 0)
                if p0 > 0 and p1 > 0:
                    signal_pnl += n0 * (p1 / p0 - 1.0)
            net_delta = equity - self.last_equity
            cost = net_delta - signal_pnl
            self.cum_signal += signal_pnl
            self._save_cum_signal(self.cum_signal)
            log.info(f"[PERP_LS] ATTRIB signal(price-only)=${signal_pnl:+.2f} net=${net_delta:+.2f} "
                     f"cost(funding+fees)=${cost:+.2f} | cum_signal=${self.cum_signal:+.2f}")

            # --- [RL_METRICS] shadow logging (additive-only; reuses vars computed above for ATTRIB) ---
            # Sortino-style risk-adjusted reward/penalty, mirroring the reward math in
            # src/quant/reward_scheme.py (RiskAdjustedReturns), computed inline here (not by
            # loading that module) purely for observability. Never affects sizing/order flow.
            rl_net_score = None
            try:
                eps = 1e-6
                rl_reward = net_delta / max(vol * equity, eps)  # risk-adjusted ratio: $ delta / $ vol (units match)
                rl_penalty = (max_dd * gross) + abs(cost)       # drawdown-scaled risk term + funding/fee cost ($)
                rl_net_score = rl_reward - rl_penalty
                log.info(f"[RL_METRICS] agent=perp_ls cycle_ts={time.time():.0f} "
                         f"reward={rl_reward:+.4f} penalty=${rl_penalty:+.2f} net_score={rl_net_score:+.2f} "
                         f"cum_signal=${self.cum_signal:+.2f}")
            except Exception as e:
                log.warning(f"[RL_METRICS] agent=perp_ls metric computation failed: {e}")

            # append this interval's signal/cost/net_score to the rolling window (legs count is
            # from the PREVIOUS cycle's order loop, since that's what actually produced this cost)
            self.edge_cost_history.append({
                "signal": signal_pnl, "cost": cost,
                "net_score": rl_net_score, "legs": self.last_legs_traded,
            })
            self._save_edge_cost_history()

            # --- rolling promotion gate (additive-only; does not affect sizing/order flow) ---
            promo_ratio, _, _, _, promo_n = self._rolling_edge_cost(PROMOTION_WINDOW)
            recent_scores = [e["net_score"] for e in self.edge_cost_history[-PROMOTION_WINDOW:]
                              if e.get("net_score") is not None]
            avg_net_score = sum(recent_scores) / len(recent_scores) if recent_scores else 0.0
            window_pass = (promo_n >= EDGE_COST_MIN_SAMPLES) and (avg_net_score > 0) and (promo_ratio > 1.0)
            self.promotion_streak = self.promotion_streak + 1 if window_pass else 0
            save_float(PROMOTION_STREAK_FILE, float(self.promotion_streak), "{:.0f}")
            promotion_ready = self.promotion_streak >= PROMOTION_CONSECUTIVE
            log.info(f"[PERP_LS] PROMOTION_READY={promotion_ready} streak={self.promotion_streak}/{PROMOTION_CONSECUTIVE} "
                     f"avg_net_score(n={promo_n})={avg_net_score:+.2f} edge_cost_ratio={promo_ratio:.2f}")

        log.info(f"[PERP_LS] equity=${equity:.2f} gross_target=${gross:.2f} "
                 f"longs={[b for b,v in tgt.items() if v>0]} shorts={[b for b,v in tgt.items() if v<0]}")

        legs_traded_this_cycle = 0
        for b in book_bases:
            px = float((tickers.get(self.sym(b)) or {}).get("last") or 0)
            if px <= 0:
                continue
            want = tgt.get(b, 0.0)
            have = cur.get(b, 0.0)
            delta = want - have
            if abs(delta) < REBAL_MIN_USD:        # dust/churn guard
                continue
            # reduce_only when shrinking magnitude or crossing through zero on the same side
            reduce_only = bool(have != 0 and (abs(want) < abs(have)) and (np.sign(want) == np.sign(have) or want == 0))
            # minimum-edge gate: never gate a reduce_only trim (risk-reducing), only opens/adds.
            # Skip if the expected $ edge (|momentum| * target notional) doesn't clear k * trailing
            # avg cost/trade -- once we have enough rolling history to trust that average.
            if not reduce_only and avg_cost_per_trade > 0:
                expected_edge = abs(mom.get(b, 0.0)) * abs(want)
                if expected_edge < MIN_EDGE_K * avg_cost_per_trade:
                    log.info(f"[PERP_LS] edge gate: skip {b} delta=${delta:+.2f} "
                             f"expected_edge=${expected_edge:.2f} < {MIN_EDGE_K}x avg_cost/trade=${avg_cost_per_trade:.2f}")
                    continue
            try:
                await self._order(b, delta, px, reduce_only=reduce_only)
                legs_traded_this_cycle += 1
            except Exception as e:
                log.error(f"[PERP_LS] order {b} failed: {e}")
        self.last_legs_traded = legs_traded_this_cycle

        # verify dollar-neutrality actually achieved (catches partial-fill imbalance)
        post = await self._positions()
        longs_usd = sum(v for v in post.values() if v > 0)
        shorts_usd = -sum(v for v in post.values() if v < 0)
        net = longs_usd - shorts_usd
        gross_now = longs_usd + shorts_usd
        skew = (net / gross_now * 100) if gross_now > 0 else 0.0
        lvl = log.warning if abs(skew) > 15 else log.info
        lvl(f"[PERP_LS] post-rebalance long=${longs_usd:.0f} short=${shorts_usd:.0f} "
            f"net=${net:+.0f} ({skew:+.1f}% of gross) -- {'SKEWED' if abs(skew)>15 else 'neutral'}")

        # neutrality REPAIR: if a leg stayed underfilled and left the book skewed, trim the heavier
        # side (reduce_only) down toward the lighter side so we never carry net directional risk.
        # reduce_only trims go WITH available liquidity, so they fill where the stuck entry couldn't.
        # tolerance band: skew under SKEW_REPAIR_PCT of gross rides to the next natural rebalance
        # instead of paying a second round of taker slippage to force-repair noise.
        if gross_now > 0 and abs(skew) <= SKEW_REPAIR_PCT:
            log.info(f"[PERP_LS] skew {skew:+.1f}% within {SKEW_REPAIR_PCT:.0f}% tolerance band -- deferring repair.")
        elif abs(net) > REPAIR_MIN_USD and gross_now > 0:
            heavy_long = net > 0
            trim = abs(net) / 2.0                      # move both sides toward the midpoint
            for b, notion in sorted(post.items(), key=lambda kv: -abs(kv[1])):
                if trim < REPAIR_MIN_USD:
                    break
                if (notion > 0) != heavy_long:         # only trim the heavier side
                    continue
                px = float((tickers.get(self.sym(b)) or {}).get("last") or 0)
                if px <= 0:
                    continue
                cut = min(trim, abs(notion) - MIN_NOTIONAL if abs(notion) > MIN_NOTIONAL else abs(notion))
                if cut < REPAIR_MIN_USD:
                    continue
                delta = -cut if notion > 0 else cut    # reduce magnitude
                try:
                    await self._order(b, delta, px, reduce_only=True)
                    trim -= cut
                except Exception as e:
                    log.error(f"[PERP_LS] neutrality trim {b} failed: {e}")
            log.info(f"[PERP_LS] neutrality repair done (was net=${net:+.0f}).")

        # snapshot the held book for next-interval attribution (positions static until next rebalance)
        self.last_notional = dict(post)
        self.last_px = {b: float((tickers.get(self.sym(b)) or {}).get("last") or 0) for b in post}
        self.last_equity = equity

    async def _flatten_reserved(self):
        """One-time reconciliation: close any position this book still holds in CARRY_RESERVED
        majors, left over from before the carve-out. Carry owns those symbols now; an orphan (e.g.
        a naked ETH long carry never adopts because ETH funding is capped) would sit unhedged and
        break neutrality. reduce_only. self._positions() filters to UNIVERSE (no majors), so this
        queries the reserved symbols directly."""
        try:
            for p in await self.hl.fetch_positions([self.sym(b) for b in CARRY_RESERVED]):
                base = p.get("symbol", "").replace("/USDC:USDC", "")
                if base not in CARRY_RESERVED:
                    continue
                amt = float(p.get("contracts", 0) or 0)
                if amt == 0:
                    continue
                side = p.get("side")
                px = float(p.get("markPrice") or p.get("entryPrice") or 0)
                if px <= 0:
                    continue
                notion = amt * px * (1 if side == "long" else -1)
                log.warning(f"[PERP_LS] reconcile: closing orphan reserved {base} "
                            f"notional=${notion:+.0f} (now carry-owned).")
                await self._order(base, -notion, px, reduce_only=True)
        except Exception as e:
            log.error(f"[PERP_LS] reserved reconcile failed: {e}")

    async def flatten(self):
        cur = await self._positions()
        tickers = await self.hl.fetch_tickers([self.sym(b) for b in cur]) if cur else {}
        for b, notion in cur.items():
            px = float((tickers.get(self.sym(b)) or {}).get("last") or 0)
            if px <= 0 or abs(notion) < 1:
                continue
            await self._order(b, -notion, px, reduce_only=True)
        log.info("[PERP_LS] flatten complete.")

    async def dry_scan(self):
        await self.rebalance()
        await self.hl.close()

    async def run_forever(self):
        log.info(f"Starting Perp L/S (LIVE={LIVE}) universe={len(UNIVERSE)} lookback={LOOKBACK_H}h "
                 f"q={QUANTILE} lev={LEVERAGE}x gross_frac={GROSS_FRACTION} dd_kill={DD_KILL}")
        if LIVE and not os.path.exists(RECONCILE_MARKER):
            await self._flatten_reserved()
            try:
                with open(RECONCILE_MARKER, "w") as f:
                    f.write("done")
            except Exception as e:
                log.error(f"[PERP_LS] reconcile marker write failed: {e}")
        try:
            while True:
                if os.path.exists("state/STALE_WAKE_HALT") and not self.halted:
                    log.error("[PERP_LS] STALE_WAKE_HALT marker present (watchdog was silent -- host likely slept). Halting until cleared.")
                    self.halted = True
                elif not os.path.exists("state/STALE_WAKE_HALT") and self.halted:
                    log.info("[PERP_LS] STALE_WAKE_HALT cleared — resuming trading.")
                    self.halted = False
                if self.halted:
                    log.error("[PERP_LS] halted by kill-switch; sleeping without trading.")
                else:
                    try:
                        await self.rebalance()
                    except Exception as e:
                        log.error(f"[PERP_LS] rebalance error: {e}")
                await asyncio.sleep(self.next_cadence)
        finally:
            await self.hl.close()


if __name__ == "__main__":
    bot = PerpLongShort()
    asyncio.run(bot.dry_scan() if not LIVE else bot.run_forever())
