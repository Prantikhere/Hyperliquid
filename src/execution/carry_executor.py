"""
Delta-neutral funding-carry executor for Hyperliquid testnet.

Strategy (validated walk-forward OOS in src/quant/funding_carry.py):
  - Universe: majors with a HL spot pair (BTC, ETH, SOL).
  - Entry: when a perp's annualized funding >= ENTRY_ANN, open a delta-neutral carry:
        LONG spot base  +  SHORT perp same base, matched USD notional, perp at 1x.
    Short perp RECEIVES funding while funding is positive; spot leg cancels price risk.
  - Hold-through exit: close only after funding stays < EXIT_ANN for EXIT_PERSIST consecutive
    checks (dodges bear-funding regimes without churning fees).
  - Perp close is reduce_only; spot close sells the base back to USDC.

SAFETY:
  - Requires USDC in BOTH the spot wallet (for the long leg) and perp wallet (for the short).
    Run scripts/split_wallet or transfer manually first; this executor does NOT move funds.
  - LEVERAGE 1x on the perp leg (a hedge, not a bet).
  - Respects MIN_NOTIONAL ($10) per leg; skips symbols it cannot size.
  - Dry-run mode (default) logs decisions without sending orders. Set CARRY_LIVE=yes to arm.

Run (read-only dry run):  venv/bin/python -m src.execution.carry_executor
Run (live testnet):       CARRY_LIVE=yes venv/bin/python -m src.execution.carry_executor
"""
import asyncio
import os
import ccxt.async_support as ccxt
from dotenv import load_dotenv
from src.utils.logger import log

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

UNIVERSE = ["BTC", "ETH", "SOL"]        # HL testnet spot pairs confirmed
HL_INTERVALS_PER_YEAR = 24 * 365        # HL funding is hourly
ENTRY_ANN = 0.15                        # enter carry when annualized funding >= 15%
ENTRY_PERSIST = 3                       # funding must stay rich for this many consecutive checks
                                        # before entering. Testnet funding spikes to garbage
                                        # (ETH 1244%, SOL +-300%) for a single interval then flips
                                        # -- a transient spike would open a carry that immediately
                                        # bleeds negative funding on exit. Persistence = only trade
                                        # funding that is actually sustained, not a one-tick artifact.
MAX_SANE_ANN = 2.0                      # >200%/yr = testnet artifact (e.g. ETH 1163%); skip, don't trade nonsense
EXIT_ANN = 0.05                         # weak-funding floor
EXIT_PERSIST = 3                        # consecutive weak checks before closing
MIN_NOTIONAL = 11.0                     # $ per leg (HL min is $10; margin buffer)
POLL_SECONDS = 3600                     # one funding interval
LIVE = os.getenv("CARRY_LIVE", "no").lower() == "yes"


class CarryExecutor:
    def __init__(self):
        self.addr = os.getenv("HL_WALLET_ADDRESS")
        self.hl = ccxt.hyperliquid({
            "privateKey": os.getenv("HL_PRIVATE") or os.getenv("HL_PRIVATE_KEY"),
            "walletAddress": self.addr,
            "options": {"defaultType": "swap"},
        })
        self.hl.set_sandbox_mode(True)
        self.hl.walletAddress = self.addr
        self.weak = {b: 0 for b in UNIVERSE}     # weak-funding streak per base (exit hold)
        self.rich = {b: 0 for b in UNIVERSE}     # rich-funding streak per base (entry hold)

    async def _ann_funding(self, base):
        fr = await self.hl.fetch_funding_rate(f"{base}/USDC:USDC")
        return float(fr.get("fundingRate") or 0.0) * HL_INTERVALS_PER_YEAR

    async def _spot_usdc(self):
        b = await self.hl.fetch_balance({"type": "spot"})
        return float(b.get("total", {}).get("USDC", 0) or 0)

    async def _perp_usdc(self):
        b = await self.hl.fetch_balance()
        return float(b.get("total", {}).get("USDC", 0) or 0)

    async def _perp_short_qty(self, base):
        """Current short size (contracts) on the perp, 0 if none/long."""
        try:
            for p in await self.hl.fetch_positions([f"{base}/USDC:USDC"]):
                amt = float(p.get("contracts", 0) or 0)
                side = p.get("side")
                if side == "short" and amt:
                    return amt
        except Exception as e:
            log.error(f"perp pos {base}: {e}")
        return 0.0

    async def _spot_base_qty(self, base):
        try:
            b = await self.hl.fetch_balance({"type": "spot"})
            return float(b.get("total", {}).get(base, 0) or 0)
        except Exception:
            return 0.0

    async def open_carry(self, base, notional):
        # Price each leg off its OWN ticker. Spot and perp reference prices can diverge
        # (badly on testnet); a perp price on a spot order trips HL's "80% away" guard.
        spot_px = (await self.hl.fetch_ticker(f"{base}/USDC"))["last"]
        perp_px = (await self.hl.fetch_ticker(f"{base}/USDC:USDC"))["last"]
        # Sanity: spot and perp of the same asset must track closely. A big gap (or a missing
        # spot price) means the spot market is broken/illiquid (HL testnet majors are fake-priced)
        # -> refuse to open a mismatched, non-hedged position.
        if not spot_px or not perp_px:
            log.error(f"[CARRY OPEN] {base} missing price spot={spot_px} perp={perp_px}. Skipping.")
            return
        if abs(spot_px - perp_px) / perp_px > 0.10:
            log.error(f"[CARRY OPEN] {base} spot/perp diverge {abs(spot_px-perp_px)/perp_px*100:.0f}% "
                      f"(spot={spot_px} perp={perp_px}) -- broken spot market, skipping.")
            return
        spot_qty = notional / spot_px
        perp_qty = notional / perp_px
        log.info(f"[CARRY OPEN] {base} notional=${notional:.2f} spot_px={spot_px} perp_px={perp_px} "
                 f"spot_qty={spot_qty:.6f} perp_qty={perp_qty:.6f} live={LIVE}")
        if not LIVE:
            return
        # Leg 1: long spot. HL market order needs a slippage bound.
        await self.hl.create_order(f"{base}/USDC", "market", "buy", spot_qty, spot_px, {"slippage": 0.05})
        # Leg 2: short perp, 1x. If this fails the spot leg already filled -> naked long, so
        # unwind the spot buy rather than sit unhedged.
        try:
            await self.hl.set_leverage(1, f"{base}/USDC:USDC")
            await self.hl.create_order(f"{base}/USDC:USDC", "market", "sell", perp_qty, perp_px, {"slippage": 0.05})
        except Exception as e:
            log.error(f"[CARRY OPEN] {base} perp short FAILED after spot filled: {e}. Unwinding spot to stay flat.")
            try:
                held = await self._spot_base_qty(base)
                if held > 0:
                    await self.hl.create_order(f"{base}/USDC", "market", "sell", held, spot_px, {"slippage": 0.05})
            except Exception as ue:
                log.error(f"[CARRY OPEN] {base} spot unwind FAILED: {ue}. NAKED LONG spot exposure -- manual check.")
            raise
        log.info(f"[CARRY OPEN OK] {base}")

    async def close_carry(self, base):
        perp_qty = await self._perp_short_qty(base)
        spot_qty = await self._spot_base_qty(base)
        perp_px = (await self.hl.fetch_ticker(f"{base}/USDC:USDC"))["last"]
        spot_px = (await self.hl.fetch_ticker(f"{base}/USDC"))["last"]
        log.info(f"[CARRY CLOSE] {base} perp_short={perp_qty} spot={spot_qty} perp_px={perp_px} spot_px={spot_px} live={LIVE}")
        if not LIVE:
            return
        if perp_qty > 0:
            await self.hl.create_order(f"{base}/USDC:USDC", "market", "buy", perp_qty, perp_px, {"reduceOnly": True, "slippage": 0.05})
        if spot_qty > 0:
            await self.hl.create_order(f"{base}/USDC", "market", "sell", spot_qty, spot_px, {"slippage": 0.05})
        log.info(f"[CARRY CLOSE OK] {base}")

    async def cycle(self):
        spot_usdc = await self._spot_usdc()
        perp_usdc = await self._perp_usdc()
        log.info(f"[CARRY] wallets spot_usdc=${spot_usdc:.2f} perp_usdc=${perp_usdc:.2f}")
        for base in UNIVERSE:
            try:
                ann = await self._ann_funding(base)
                open_now = await self._perp_short_qty(base) > 0
                log.info(f"[CARRY] {base} ann_funding={ann*100:.1f}% in_pos={open_now} weak={self.weak[base]}")
                if not open_now:
                    # track how long funding has stayed in the sane, rich band [ENTRY_ANN, MAX_SANE_ANN)
                    if ENTRY_ANN <= ann < MAX_SANE_ANN:
                        self.rich[base] += 1
                    else:
                        self.rich[base] = 0
                    if ann >= MAX_SANE_ANN:
                        log.warning(f"[CARRY] {base} funding {ann*100:.0f}% exceeds sane cap "
                                    f"{MAX_SANE_ANN*100:.0f}% -- treating as testnet artifact, skipping.")
                    elif self.rich[base] >= ENTRY_PERSIST:
                        notional = min(spot_usdc, perp_usdc)
                        if notional >= MIN_NOTIONAL:
                            await self.open_carry(base, min(notional, MIN_NOTIONAL * 3))
                            self.rich[base] = 0
                        else:
                            log.warning(f"[CARRY] {base} funding rich ({ann*100:.1f}%) but notional ${notional:.2f} < ${MIN_NOTIONAL}. "
                                        f"Fund the spot wallet to enable.")
                    elif ann >= ENTRY_ANN:
                        log.info(f"[CARRY] {base} funding rich ({ann*100:.1f}%) streak={self.rich[base]}/{ENTRY_PERSIST} "
                                 f"-- waiting for persistence before entry.")
                    self.weak[base] = 0
                else:
                    self.weak[base] = self.weak[base] + 1 if ann < EXIT_ANN else 0
                    if self.weak[base] >= EXIT_PERSIST:
                        await self.close_carry(base)
                        self.weak[base] = 0
            except Exception as e:
                log.error(f"[CARRY] {base} cycle error: {e}")

    async def run_forever(self):
        log.info(f"Starting Carry Executor (LIVE={LIVE}) universe={UNIVERSE} "
                 f"entry>={ENTRY_ANN*100:.0f}% exit<{EXIT_ANN*100:.0f}% persist={EXIT_PERSIST}")
        try:
            while True:
                await self.cycle()
                await asyncio.sleep(POLL_SECONDS)
        finally:
            await self.hl.close()

    async def dry_scan(self):
        """One read-only pass: show live funding + what it would do. No orders."""
        await self.cycle()
        await self.hl.close()


if __name__ == "__main__":
    asyncio.run(CarryExecutor().dry_scan() if not LIVE else CarryExecutor().run_forever())
