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
            
            # Live L2 orderbook snapshot
            l2 = None
            bids, asks = [], []
            try:
                l2 = self.hl_sdk.info.l2_snapshot(coin)
                bids = l2.get("levels", [[]])[0]
                asks = l2.get("levels", [[], []])[1] if len(l2.get("levels", [])) > 1 else []
            except Exception as e:
                log.warning(f"[HL_L2] Failed to get snapshot for {coin}: {e}")

            # Check spread on entries: prevent entering on wide/illiquid spreads
            if not reduce_only and bids and asks:
                best_bid = float(bids[0]["px"])
                best_ask = float(asks[0]["px"])
                if best_bid > 0:
                    spread_pct = (best_ask - best_bid) / best_bid
                    if spread_pct > 0.0020:  # 0.20% max allowed spread for entry
                        log.warning(f"[SPREAD_GUARD] {coin} entry blocked: spread too wide ({spread_pct*100:.2f}% > 0.20%)")
                        return {"error": f"Spread too wide: {spread_pct*100:.2f}% > 0.20%"}

            oracle_px = self.hl_sdk._get_oracle_px(coin)
            if oracle_px <= 0:
                return {"error": f"Cannot get oracle price for {coin}"}
            
            if reduce_only:
                # Exiting open position: cross spread to best bid/ask
                if is_buy and asks:
                    limit_px = float(asks[0]["px"])
                elif not is_buy and bids:
                    limit_px = float(bids[0]["px"])
                else:
                    limit_px = oracle_px * (1.005 if is_buy else 0.995)
                # Clamp within 2.5% oracle band
                limit_px = max(oracle_px * 0.975, min(oracle_px * 1.025, limit_px))
            else:
                # Entering new position: match immediate liquidity
                if is_buy and asks:
                    limit_px = float(asks[0]["px"])
                elif not is_buy and bids:
                    limit_px = float(bids[0]["px"])
                else:
                    limit_px = oracle_px * (1.002 if is_buy else 0.998)
                limit_px = max(oracle_px * 0.98, min(oracle_px * 1.02, limit_px))
            
            order_res = self.hl_sdk.place_limit_order(
                coin, is_buy, quantity, limit_px, reduce_only=reduce_only, tif="Gtc"
            )

            # Prevent stale resting entry orders: if entry order rests, wait 2.5s; if still unfilled, cancel!
            if not reduce_only and order_res and order_res.get("id"):
                oid = order_res.get("id")
                if order_res.get("status") == "resting":
                    await asyncio.sleep(2.5)
                    try:
                        open_orders = self.hl_sdk.get_open_orders()
                        still_open = any(str(o.get("oid")) == str(oid) for o in open_orders)
                        if still_open:
                            log.warning(f"[HL_EXEC] Entry order {oid} for {coin} did not fill within 2.5s. Cancelling resting order.")
                            self.hl_sdk.cancel_order(coin, int(oid))
                            return {"error": f"Entry order {oid} rested and was canceled to prevent stale adverse fill", "status": "UNFILLED"}
                        else:
                            order_res["status"] = "filled"
                    except Exception as ex_chk:
                        log.warning(f"[HL_EXEC] Check resting order error: {ex_chk}")

            return order_res

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
