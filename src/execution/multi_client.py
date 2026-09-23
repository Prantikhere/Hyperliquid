import ccxt.async_support as ccxt
import asyncio
import os
from src.utils.logger import log
from src.execution.hl_raw import HlSdkClient

class MultiExchangeClient:
    def __init__(self):
        # 1. BingX Perpetual (USDT-M)
        self.bingx = ccxt.bingx({
            'apiKey': os.getenv("BINGX_API_KEY"),
            'secret': os.getenv("BINGX_SECRET_KEY"),
            'options': {'defaultType': 'swap'},
            'timeout': 30000,
        })
        
        # 2. Hyperliquid — using official SDK (bypasses ccxt async issues)
        self.hl_sdk = HlSdkClient()
        
        # Keep ccxt hl for reference/fallback only
        self.hl = None
        
        self.exchanges = {
            "bingx": self.bingx,
            "hyperliquid": None  # SDK handles HL
        }

    def _get_mapped_symbol(self, exchange_id, symbol):
        if exchange_id == "hyperliquid":
            return symbol.replace("/USDT", "")
        return symbol

    async def set_leverage(self, exchange_id, symbol, leverage):
        if exchange_id == "hyperliquid":
            coin = self._get_mapped_symbol(exchange_id, symbol)
            return self.hl_sdk.set_leverage(coin, int(leverage))
        try:
            exchange = self.exchanges.get(exchange_id)
            if exchange:
                mapped_symbol = self._get_mapped_symbol(exchange_id, symbol)
                return await exchange.set_leverage(int(leverage), mapped_symbol)
        except Exception as e:
            log.error(f"Error setting leverage on {exchange_id} for {symbol}: {e}")

    async def place_order(self, exchange_id, symbol, side, order_type, quantity, price=None, leverage=None, reduce_only=False):
        if exchange_id == "hyperliquid":
            coin = self._get_mapped_symbol(exchange_id, symbol)
            is_buy = side.lower() == "buy"
            
            # Set leverage for new entries
            if not reduce_only:
                lev = int(leverage) if leverage else 5
                self.hl_sdk.set_leverage(coin, lev)
            
            # HL testnet: IOC market orders fail (no resting orders to match)
            # Use GTC limit orders at oracle price for ALL HL orders
            oracle_px = self.hl_sdk._get_oracle_px(coin)
            if oracle_px <= 0:
                return {"error": f"Cannot get oracle price for {coin}"}
            
            # For exits (reduce_only), use oracle price directly
            # For entries, add small slippage in the right direction
            if reduce_only:
                limit_px = oracle_px  # Exit at oracle
            elif is_buy:
                limit_px = oracle_px * 1.005  # Buy slightly above oracle
            else:
                limit_px = oracle_px * 0.995  # Sell slightly below oracle
            
            return self.hl_sdk.place_limit_order(
                coin, is_buy, quantity, limit_px, reduce_only=reduce_only, tif="Gtc"
            )

        # BingX uses ccxt
        try:
            exchange = self.exchanges.get(exchange_id)
            if not exchange: return None
            
            mapped_symbol = self._get_mapped_symbol(exchange_id, symbol)
            ccxt_side = side.lower()
            
            if leverage is None:
                leverage = 7
            await self.set_leverage(exchange_id, symbol, leverage)
            
            log.info(f"[{exchange_id}] Sending {ccxt_side} {order_type} for {quantity} {mapped_symbol} (Leverage: {leverage}x, ReduceOnly: {reduce_only})")
            
            params = {}
            if reduce_only:
                params['reduceOnly'] = True
            
            if order_type.upper() == "MARKET":
                return await exchange.create_order(mapped_symbol, 'market', ccxt_side, quantity, price, params)
            else:
                return await exchange.create_order(mapped_symbol, 'limit', ccxt_side, quantity, price, params)
                
        except Exception as e:
            log.error(f"API CALL ERROR on {exchange_id} ({symbol}): {e}")
            return {"error": str(e)}

    async def get_balance(self, exchange_id):
        if exchange_id == "hyperliquid":
            return self.hl_sdk.get_balance()
        try:
            exchange = self.exchanges.get(exchange_id)
            if exchange:
                return await exchange.fetch_balance()
        except Exception as e:
            log.error(f"Error fetching balance on {exchange_id}: {e}")
            return {}

    async def get_onchain_positions(self, exchange_id):
        """Fetch positions directly from the exchange chain/API."""
        if exchange_id == "hyperliquid":
            return self.hl_sdk.get_positions()
        try:
            exchange = self.exchanges.get(exchange_id)
            if exchange:
                return await exchange.fetch_positions()
            return []
        except Exception as e:
            log.error(f"Error fetching on-chain positions for {exchange_id}: {e}")
            raise e

    async def close(self):
        if self.bingx:
            await self.bingx.close()
        if self.hl_sdk:
            self.hl_sdk.close()
