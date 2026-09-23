"""
Raw Hyperliquid client using the official HL Python SDK.
Bypasses ccxt async issues entirely.
"""
import os
import asyncio
import logging
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

from eth_account import Account
from hyperliquid.exchange import Exchange as HlExchange
from hyperliquid.info import Info as HlInfo
from hyperliquid.utils import constants

log = logging.getLogger(__name__)

class HlSdkClient:
    """Wrapper around hyperliquid-python-sdk for order placement and queries."""

    def __init__(self):
        private_key = os.getenv("HL_PRIVATE") or os.getenv("HL_PRIVATE_KEY")
        self.wallet_address = os.getenv("HL_WALLET_ADDRESS")
        
        wallet = Account.from_key(private_key)
        self.exchange = HlExchange(wallet, constants.TESTNET_API_URL)
        self.info = HlInfo(constants.TESTNET_API_URL, skip_ws=True)
        
        # Cache coin-to-asset mapping and szDecimals
        meta = self.info.meta()
        self.coin_to_asset = {a["name"]: i for i, a in enumerate(meta["universe"])}
        self.coin_sz_decimals = {a["name"]: a.get("szDecimals", 0) for a in meta["universe"]}
        log.info(f"[HL-SDK] Initialized. Coins: {len(self.coin_to_asset)}")

    def _get_oracle_px(self, coin: str) -> float:
        """Get current mid price from allMids (used as proxy for oracle)."""
        mids = self.info.all_mids()
        return float(mids.get(coin, 0))

    def _round_price(self, coin: str, price: float) -> float:
        """Round price to valid tick size and sig figs for HL wire without float_to_wire errors."""
        sz_dec = self.coin_sz_decimals.get(coin, 0)
        max_decimals = max(0, 6 - sz_dec)
        formatted = f"{price:.5g}"
        val = round(float(formatted), min(max_decimals, 5))
        return round(val, 6)

    def _round_qty(self, coin: str, qty: float) -> float:
        """Round quantity to valid szDecimals for the asset."""
        decimals = self.coin_sz_decimals.get(coin, 0)
        rounded = round(qty, decimals)
        log.debug(f"[HL-SDK] {coin} qty rounded: {qty} -> {rounded} (szDecimals={decimals})")
        return rounded

    def place_market_order(self, coin: str, is_buy: bool, qty: float,
                           slippage: float = 0.01, reduce_only: bool = False) -> dict:
        """Place a market order via IOC limit at oracle + slippage."""
        oracle_px = self._get_oracle_px(coin)
        if oracle_px <= 0:
            return {"error": f"Cannot get price for {coin}"}

        # Round quantity to valid szDecimals
        qty = self._round_qty(coin, qty)
        if qty <= 0:
            return {"error": f"Quantity too small for {coin} after rounding"}

        # Calculate IOC price: buy slightly above oracle, sell slightly below
        if is_buy:
            ioc_px = self._round_price(coin, oracle_px * (1 + slippage))
        else:
            ioc_px = self._round_price(coin, oracle_px * (1 - slippage))

        log.info(f"[HL-SDK] {coin} {'BUY' if is_buy else 'SELL'} {qty} @ {ioc_px} (oracle={oracle_px}, slippage={slippage})")

        try:
            result = self.exchange.order(
                coin, is_buy, qty, ioc_px,
                order_type={"limit": {"tif": "Ioc"}},
                reduce_only=reduce_only
            )
            status = result.get("response", {}).get("data", {}).get("statuses", [{}])[0]
            if "error" in status:
                log.error(f"[HL-SDK] Order error: {status['error']}")
                return {"error": status["error"]}
            elif "resting" in status:
                oid = status["resting"]["oid"]
                log.info(f"[HL-SDK] Order resting: oid={oid}")
                return {"id": oid, "status": "resting"}
            elif "filled" in status:
                filled = status["filled"]
                log.info(f"[HL-SDK] Order filled: avg_px={filled.get('avgPx')} sz={filled.get('sz')}")
                return {"id": filled.get("oid", ""), "status": "filled",
                        "avg_price": filled.get("avgPx")}
            return {"id": "", "status": str(status)}
        except Exception as e:
            log.error(f"[HL-SDK] Order exception: {e}")
            return {"error": str(e)}

    def place_limit_order(self, coin: str, is_buy: bool, qty: float,
                          price: float, reduce_only: bool = False,
                          tif: str = "Gtc") -> dict:
        """Place a limit order."""
        rounded_px = self._round_price(coin, price)
        qty = self._round_qty(coin, qty)
        if qty <= 0:
            return {"error": f"Quantity too small for {coin} after rounding"}
        log.info(f"[HL-SDK] {coin} LIMIT {'BUY' if is_buy else 'SELL'} {qty} @ {rounded_px} (tif={tif})")

        try:
            result = self.exchange.order(
                coin, is_buy, qty, rounded_px,
                order_type={"limit": {"tif": tif}},
                reduce_only=reduce_only
            )
            status = result.get("response", {}).get("data", {}).get("statuses", [{}])[0]
            if "error" in status:
                log.error(f"[HL-SDK] Limit order error: {status['error']}")
                return {"error": status["error"]}
            elif "resting" in status:
                oid = status["resting"]["oid"]
                log.info(f"[HL-SDK] Limit resting: oid={oid}")
                return {"id": oid, "status": "resting"}
            elif "filled" in status:
                filled = status["filled"]
                return {"id": filled.get("oid", ""), "status": "filled",
                        "avg_price": filled.get("avgPx")}
            return {"id": "", "status": str(status)}
        except Exception as e:
            log.error(f"[HL-SDK] Limit order exception: {e}")
            return {"error": str(e)}

    def cancel_order(self, coin: str, oid: int) -> dict:
        """Cancel an open order."""
        try:
            result = self.exchange.cancel(coin, oid)
            return result
        except Exception as e:
            log.error(f"[HL-SDK] Cancel error: {e}")
            return {"error": str(e)}

    def set_leverage(self, coin: str, leverage: int) -> dict:
        """Set leverage for a coin."""
        try:
            result = self.exchange.update_leverage(leverage, coin, is_cross=True)
            return result
        except Exception as e:
            log.error(f"[HL-SDK] Set leverage error: {e}")
            return {"error": str(e)}

    def get_positions(self) -> list:
        """Get all open positions."""
        try:
            state = self.info.user_state(self.wallet_address)
            return state.get("assetPositions", [])
        except Exception as e:
            log.error(f"[HL-SDK] Get positions error: {e}")
            return []

    def get_open_orders(self) -> list:
        """Get all open orders."""
        try:
            return self.info.open_orders(self.wallet_address)
        except Exception as e:
            log.error(f"[HL-SDK] Get open orders error: {e}")
            return []

    def get_position(self, coin: str) -> Optional[dict]:
        """Get position for a specific coin."""
        for pos in self.get_positions():
            p = pos.get("position", {})
            if p.get("coin") == coin:
                return p
        return None

    def get_balance(self) -> dict:
        """Get account balance info."""
        try:
            state = self.info.user_state(self.wallet_address)
            margin = state.get("marginSummary", {})
            return {
                "account_value": float(margin.get("accountValue", 0)),
                "margin_used": float(margin.get("totalMarginUsed", 0)),
                "available": float(margin.get("accountValue", 0)) - float(margin.get("totalMarginUsed", 0)),
            }
        except Exception as e:
            log.error(f"[HL-SDK] Get balance error: {e}")
            return {"account_value": 0, "margin_used": 0, "available": 0}

    def close(self):
        """No persistent resources to close."""
        pass
