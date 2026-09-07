import ccxt.async_support as ccxt
import os
from src.utils.logger import log

class MultiExchangeClient:
    def __init__(self):
        # 1. BingX Perpetual (USDT-M)
        self.bingx = ccxt.bingx({
            'apiKey': os.getenv("BINGX_API_KEY"),
            'secret': os.getenv("BINGX_SECRET_KEY"),
            'options': {'defaultType': 'swap'}
        })
        
        # 2. Hyperliquid API Agent Integration.
        # Signing key and wallet address are both taken from .env and MUST belong to the same
        # account or Hyperliquid rejects the order. Env-var aliases tolerate either name.
        self.primary_addr = os.getenv("HL_WALLET_ADDRESS")
        hl_key = os.getenv("HL_PRIVATE") or os.getenv("HL_PRIVATE_KEY")
        self.hl = ccxt.hyperliquid({
            'privateKey': hl_key,
            'walletAddress': self.primary_addr,
            'options': {
                'defaultType': 'swap',
                'slippage': 0.05
            }
        })
        self.hl.set_sandbox_mode(True)
        self.hl.walletAddress = self.primary_addr
        
        self.exchanges = {
            "bingx": self.bingx,
            "hyperliquid": self.hl
        }

    def _get_mapped_symbol(self, exchange_id, symbol):
        if exchange_id == "hyperliquid":
            # Per user request: HL uses USDC
            return symbol.replace("/USDT", "/USDC:USDC")
        return symbol

    async def set_leverage(self, exchange_id, symbol, leverage):
        try:
            exchange = self.exchanges.get(exchange_id)
            if exchange:
                mapped_symbol = self._get_mapped_symbol(exchange_id, symbol)
                # Hyperliquid requires setting leverage before trading
                return await exchange.set_leverage(int(leverage), mapped_symbol)
        except Exception as e:
            log.error(f"Error setting leverage on {exchange_id} for {symbol}: {e}")

    async def place_order(self, exchange_id, symbol, side, order_type, quantity, price=None, leverage=None, reduce_only=False):
        try:
            exchange = self.exchanges.get(exchange_id)
            if not exchange: return None
            
            mapped_symbol = self._get_mapped_symbol(exchange_id, symbol)
            ccxt_side = side.lower()
            
            # Enforce Leverage per User Request / dynamic calculation
            if leverage is None:
                leverage = 5 if exchange_id == "hyperliquid" else 7
            await self.set_leverage(exchange_id, symbol, leverage)
            
            log.info(f"[{exchange_id}] Sending {ccxt_side} {order_type} for {quantity} {mapped_symbol} (Leverage: {leverage}x, ReduceOnly: {reduce_only})")
            
            params = {}
            if reduce_only:
                params['reduceOnly'] = True
            if exchange_id == "hyperliquid" and order_type.upper() == "MARKET" and price:
                params['slippage'] = 0.05
            
            if order_type.upper() == "MARKET":
                return await exchange.create_order(mapped_symbol, 'market', ccxt_side, quantity, price, params)
            else:
                return await exchange.create_order(mapped_symbol, 'limit', ccxt_side, quantity, price, params)
                
        except Exception as e:
            log.error(f"API CALL ERROR on {exchange_id} ({symbol}): {e}")
            return {"error": str(e)}

    async def get_balance(self, exchange_id):
        try:
            exchange = self.exchanges.get(exchange_id)
            if exchange:
                return await exchange.fetch_balance()
        except Exception as e:
            log.error(f"Error fetching balance on {exchange_id}: {e}")
            return {}

    async def get_onchain_positions(self, exchange_id):
        """Fetch positions directly from the exchange chain/API."""
        try:
            exchange = self.exchanges.get(exchange_id)
            if exchange:
                return await exchange.fetch_positions()
            return []
        except Exception as e:
            log.error(f"Error fetching on-chain positions for {exchange_id}: {e}")
            raise e

    async def close(self):
        for exchange in self.exchanges.values():
            await exchange.close()
