from src.execution.multi_client import MultiExchangeClient
from src.utils.db import DatabaseManager
from src.utils.logger import log
import os
import json
import asyncio

class ExecutionAgent:
    def __init__(self, db=None):
        self.db = db or DatabaseManager()
        self.multi_client = MultiExchangeClient()
        self.global_mode = os.getenv("TRADING_MODE", "paper")

    def _live_safe(self):
        """Live orders require BOTH a backtest-proven config (live_safe) and an
        explicit operator arm flag. Prevents deploying a losing strategy with real funds."""
        armed = os.getenv("LIVE_TRADING_ARMED", "no").lower() == "yes"
        proven = False
        try:
            with open("models_local/strategy_config.json") as f:
                proven = bool(json.load(f).get("live_safe", False))
        except Exception:
            proven = False
        return armed and proven

    async def execute_trade(self, trade_params):
        """Execute trade and log decision metadata for explainability."""
        symbol = trade_params['symbol']
        side = trade_params['side']
        quantity = trade_params['quantity']
        price = trade_params.get('price', 0)
        exchange_id = trade_params.get('target_exchange', 'bingx')
        # Lineage metadata
        metadata = trade_params.get('metadata', {})
        
        # Use specific exchange mode if set, else fallback to global mode
        mode = os.getenv(f"{exchange_id.upper()}_MODE", self.global_mode)
        
        log.info(f"[{exchange_id.upper()}] Execution: {side} {quantity} {symbol}")

        reduce_only = trade_params.get('reduce_only', False)
        # Gate NEW live positions on backtest proof + arm flag. Exits (reduce_only) always allowed.
        if mode == "live" and not reduce_only and not self._live_safe():
            log.warning(f"[{exchange_id}] LIVE entry BLOCKED for {symbol}: strategy not backtest-proven (live_safe) or not armed. Simulating instead.")
            return {"status": "BLOCKED", "reason": "not live_safe / not armed"}

        if mode == "live":
            order_type = trade_params.get('order_type', 'MARKET')
            log.info(f"[{exchange_id}] Sending LIVE {order_type} Order at {price}...")
            leverage = trade_params.get('leverage')
            reduce_only = trade_params.get('reduce_only', False)
            res = await self.multi_client.place_order(exchange_id, symbol, side, order_type, quantity, price, leverage, reduce_only=reduce_only)
            
            if isinstance(res, dict) and "error" in res:
                # Log failure to DB for audit
                self.db.insert_trade(symbol, exchange_id, side, price, quantity, f"FAILED: {res['error']}", metadata)
                return {"status": "ERROR", "msg": res["error"]}
                
            # SUCCESS: Log actual on-chain event
            log.info(f"[{exchange_id}] LIVE ORDER SUCCESSFUL: {res.get('id', 'N/A')}")
            self.db.insert_trade(symbol, exchange_id, side, price, quantity, "LIVE_OK", metadata)
            # Update position locally as an immediate cache; reconciliation agent will verify this.
            self.db.update_position(symbol, exchange_id, price, quantity if side == "BUY" else -quantity)
            return {"status": "OK", "details": res}
        
        # Paper/Simulated: Log to stdout but NOT to database per user "no fictitious activity" policy.
        log.warning(f"[{exchange_id}] SIMULATED TRADE (Not logged to DB): {side} {quantity} {symbol}")
        return {"status": "SIMULATED", "exchange": exchange_id}
