"""
One-shot USDC mover: perp wallet -> spot wallet on Hyperliquid testnet.

funding_carry (src/execution/carry_executor.py) needs USDC in BOTH the spot wallet
(long leg) and the perp wallet (short leg). The perp wallet already holds testnet
USDC (shared margin pool with hl_perp_ls.py / hl_executor.py); the spot wallet is
empty. This script performs the internal HL "usdClassTransfer" action (exposed by
ccxt.hyperliquid as the unified `transfer()` method) to move a slice of that balance
from perp -> spot.

SAFETY:
  - Dry-run by default: prints current balances and what it WOULD transfer.
  - Set --live or SPLIT_WALLET_LIVE=yes to actually send the transfer.
  - Does NOT touch hl_perp_ls.py / hl_executor.py; only moves USDC between the two
    wallets of the SAME account. Leaves ~$141 in perp margin at the $60 default.

Run (dry run):  venv/bin/python -m scripts.split_wallet
Run (live):     venv/bin/python -m scripts.split_wallet --live
                SPLIT_WALLET_LIVE=yes venv/bin/python -m scripts.split_wallet
"""
import argparse
import asyncio
import os
import sys

import ccxt.async_support as ccxt
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.utils.logger import log

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

AMOUNT_USDC = 60.0  # perp -> spot; leaves ~$141 perp margin for hl_perp_ls/hl_executor

LIVE_ENV = os.getenv("SPLIT_WALLET_LIVE", "no").lower() == "yes"


def make_client():
    addr = os.getenv("HL_WALLET_ADDRESS")
    hl = ccxt.hyperliquid({
        "privateKey": os.getenv("HL_PRIVATE") or os.getenv("HL_PRIVATE_KEY"),
        "walletAddress": addr,
        "options": {"defaultType": "swap"},
    })
    hl.set_sandbox_mode(True)
    hl.walletAddress = addr
    return hl


async def spot_usdc(hl):
    b = await hl.fetch_balance({"type": "spot"})
    return float(b.get("total", {}).get("USDC", 0) or 0)


async def perp_usdc(hl):
    b = await hl.fetch_balance()
    return float(b.get("total", {}).get("USDC", 0) or 0)


async def main(amount, live):
    hl = make_client()
    try:
        spot_before = await spot_usdc(hl)
        perp_before = await perp_usdc(hl)
        log.info(f"[SPLIT] current balances: spot_usdc=${spot_before:.2f} perp_usdc=${perp_before:.2f}")
        log.info(f"[SPLIT] plan: transfer ${amount:.2f} USDC perp -> spot (live={live})")

        if not live:
            log.info("[SPLIT] dry-run only, no transfer sent. Use --live or SPLIT_WALLET_LIVE=yes to arm.")
            return

        if amount > perp_before:
            log.error(f"[SPLIT] requested ${amount:.2f} exceeds perp balance ${perp_before:.2f}. Aborting.")
            return

        resp = await hl.transfer("USDC", amount, "swap", "spot")
        log.info(f"[SPLIT] transfer response: {resp}")

        spot_after = await spot_usdc(hl)
        perp_after = await perp_usdc(hl)
        log.info(f"[SPLIT] balances after transfer: spot_usdc=${spot_after:.2f} perp_usdc=${perp_after:.2f}")
    finally:
        await hl.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("amount", nargs="?", type=float, default=AMOUNT_USDC,
                         help=f"USDC amount to move perp -> spot (default ${AMOUNT_USDC:.2f})")
    parser.add_argument("--live", action="store_true", help="actually send the transfer")
    args = parser.parse_args()
    asyncio.run(main(args.amount, args.live or LIVE_ENV))
