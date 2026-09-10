import asyncio
import os
import json
from src.utils.logger import log
from src.utils.db import DatabaseManager
from src.agents.execution_agent import ExecutionAgent
from src.quant.regime_engine import RegimeEngine
from src.quant.multi_strategy import StrategyEnsemble
import redis

class SettlementAgent:
    """
    Dedicated agent for continuous profit booking and position management.
    Ensures that once a trade is made, it is actively managed to book profit.
    """
    def __init__(self, owned_symbols=None):
        self.db = DatabaseManager()
        self.execution_agent = ExecutionAgent(self.db)
        self.redis = redis.Redis(host=os.getenv('REDIS_HOST', 'localhost'), port=6379, decode_responses=True)
        self.regime_engine = RegimeEngine()
        self.strategy_ensemble = StrategyEnsemble()
        # Restrict profit-booking to this bot's own universe. Without this filter,
        # run_settlement_cycle scans every position in the DB -- including legs
        # opened by perp_ls/pairs_arb (same wallet) -- and force-closes them on
        # its own TP/SL/reversion logic, silently untracked by those bots' own logs.
        self.owned_symbols = set(owned_symbols) if owned_symbols is not None else None

    async def run_settlement_cycle(self, exchange_id="hyperliquid"):
        """Scan all open positions for profit booking opportunities."""
        try:
            positions = self.db.get_positions()
            
            # Fetch open orders once for the exchange to avoid rate limiting and allow duplicate checks
            try:
                open_orders = await self.execution_agent.multi_client.exchanges[exchange_id].fetch_open_orders()
            except Exception as e:
                log.error(f"Failed to fetch open orders for {exchange_id} duplicate check: {e}")
                open_orders = []

            for (symbol, eid), pos in positions.items():
                if eid != exchange_id or pos['quantity'] == 0:
                    continue
                if self.owned_symbols is not None and symbol not in self.owned_symbols:
                    continue

                # Get real-time price
                current_price_str = self.redis.get(f"price:{exchange_id}:{symbol}") or self.redis.get(f"price:{symbol}")
                if not current_price_str:
                    continue
                
                current_price = float(current_price_str)
                avg_price = pos['avg_price']
                quantity = pos['quantity']
                
                # Fetch recent prices for dynamic volatility
                # Reverse newest-first query result to chronological order for correct volatility/regime math.
                prices_raw = self.db.execute_query("SELECT price FROM external_prices WHERE symbol LIKE %s ORDER BY time DESC LIMIT 100", (f"%{symbol}%",))
                price_list = [p[0] for p in prices_raw][::-1] if prices_raw else [current_price]
                
                # Dynamic Volatility TP/SL Scaling (protective backstops)
                tp_threshold, sl_threshold = self.regime_engine.get_dynamic_thresholds(price_list)

                # Primary exit for the mean-reversion edge: exit once the signal that opened the
                # trade has reverted to neutral (0.5). Mirrors backtester.simulate exactly so live
                # behaviour matches the backtested/validated exit. Long opened on a bullish (>0.5)
                # signal -> exit when composite fades to <=0.5; short exits when it rises to >=0.5.
                regime = self.regime_engine.detect_regime(price_list)
                sig = self.strategy_ensemble.get_signals(price_list)
                comp = self.strategy_ensemble.composite_score(sig, regime)

                # Calculate ROI
                # (Price - Entry) / Entry
                side = "LONG" if quantity > 0 else "SHORT"
                if side == "LONG":
                    roi = (current_price - avg_price) / avg_price if avg_price > 0 else 0
                else:
                    roi = (avg_price - current_price) / avg_price if avg_price > 0 else 0

                # Trend-following positions ride to TP/SL; only mean-reversion / neutral trades
                # book at reversion-to-mean. Mirrors backtester.simulate exit model.
                # Reversion exit must clear round-trip fees with margin (mirrors backtester
                # REVERT_MIN_ROI). Booking a reversion below ~0.15% ROI is a net loss after fees.
                from src.quant.backtester import REVERT_MIN_ROI
                is_trend = "TRENDING" in (regime or "").upper()

                # Trailing profit: if ROI has been higher than current, lock in some gains.
                # Track peak ROI per symbol in redis; if current ROI has fallen >30% from peak,
                # exit to lock in the remaining profit instead of riding back to zero.
                # TTL extended to 4h to survive slow trade durations; peak threshold lowered to 0.5%
                # so even small mean-reversion gains get trail-protected.
                peak_key = f"peak_roi:{exchange_id}:{symbol}"
                peak_roi = float(self.redis.get(peak_key) or 0)
                if roi > peak_roi:
                    self.redis.set(peak_key, roi, ex=14400)  # 4h TTL
                    peak_roi = roi
                trailing_exit = (roi > 0.005) and (peak_roi > 0.005) and (roi < peak_roi * 0.7)

                reverted = (not is_trend) and (roi > REVERT_MIN_ROI) and (
                    (side == "LONG" and comp <= 0.5) or (side == "SHORT" and comp >= 0.5)
                )

                # MARKET TREND-BASED EXITS (NEW LOGIC)
                # 1. Regime change exit: exit when regime becomes unfavorable
                regime_unfavorable = False
                if side == "LONG" and "HIGH_VOL" in (regime or "").upper():
                    regime_unfavorable = True  # LONG in high vol = risky
                elif side == "SHORT" and "TRENDING" in (regime or "").upper():
                    regime_unfavorable = True  # SHORT in uptrend = risky

                # 2. Momentum fade exit: exit when composite drops significantly
                momentum_fading = (roi > 0.02) and (comp < 0.35)  # In profit but signal weak

                # 3. Time-based exit: exit if held too long without progress (4 hours)
                import time
                position_age_key = f"position_age:{exchange_id}:{symbol}"
                position_age = float(self.redis.get(position_age_key) or 0)
                if position_age == 0:
                    self.redis.set(position_age_key, time.time(), ex=86400)  # 24h TTL
                    position_age = time.time()
                held_hours = (time.time() - position_age) / 3600
                stale_position = (held_hours > 4) and (roi < 0.05)  # Held 4h+ with <5% gain

                log.info(f"[SETTLEMENT] {exchange_id} {symbol} {side} | ROI: {roi*100:.2f}% | Peak: {peak_roi*100:.2f}% | TP: {tp_threshold*100:.2f}% | SL: {sl_threshold*100:.2f}% | comp: {comp:.2f} | regime: {regime} | held: {held_hours:.1f}h")

                # Action Logic
                decision = None
                order_type = "MARKET"
                if roi >= tp_threshold:
                    log.info(f"$$$ PROFIT TARGET HIT for {symbol} on {exchange_id}. Booking Profit.")
                    decision = "SELL" if side == "LONG" else "BUY"
                    order_type = "MARKET"
                elif roi <= sl_threshold:
                    log.info(f"!!! STOP LOSS HIT for {symbol} on {exchange_id}. Cutting Loss.")
                    decision = "SELL" if side == "LONG" else "BUY"
                    order_type = "MARKET" # Use market order to guarantee immediate exit on SL
                elif trailing_exit:
                    log.info(f"~~~ TRAILING EXIT for {symbol} on {exchange_id} (ROI={roi*100:.2f}% from peak={peak_roi*100:.2f}%). Locking profit.")
                    decision = "SELL" if side == "LONG" else "BUY"
                    order_type = "MARKET"
                elif reverted and roi > 0:
                    # Signal reverted to mean while in profit: book the reversion (the validated edge).
                    log.info(f"~~~ REVERSION EXIT for {symbol} on {exchange_id} (comp={comp:.2f}). Booking reversion profit.")
                    decision = "SELL" if side == "LONG" else "BUY"
                    order_type = "MARKET"
                elif regime_unfavorable and roi > 0:
                    # Market regime became unfavorable while in profit
                    log.info(f"~~~ REGIME EXIT for {symbol} on {exchange_id} (regime={regime}). Booking profit before regime change impact.")
                    decision = "SELL" if side == "LONG" else "BUY"
                    order_type = "MARKET"
                elif momentum_fading:
                    # Signal momentum fading while in profit
                    log.info(f"~~~ MOMENTUM EXIT for {symbol} on {exchange_id} (comp={comp:.2f}). Booking profit before momentum dies.")
                    decision = "SELL" if side == "LONG" else "BUY"
                    order_type = "MARKET"
                elif stale_position:
                    # Position held too long without progress
                    log.info(f"~~~ STALE EXIT for {symbol} on {exchange_id} (held={held_hours:.1f}h, ROI={roi*100:.2f}%). Cutting stale position.")
                    decision = "SELL" if side == "LONG" else "BUY"
                    order_type = "MARKET"

                if decision:
                    mapped_symbol = self.execution_agent.multi_client._get_mapped_symbol(exchange_id, symbol)
                    
                    # Instead of skipping, cancel any existing stale open orders for this symbol
                    for order in open_orders:
                        if order.get('symbol') == mapped_symbol:
                            try:
                                log.info(f"[SETTLEMENT] Canceling stale open order {order.get('id')} for {symbol} on {exchange_id}...")
                                await self.execution_agent.multi_client.exchanges[exchange_id].cancel_order(order.get('id'), mapped_symbol)
                            except Exception as ex_cancel:
                                log.error(f"Failed to cancel order {order.get('id')} on {exchange_id}: {ex_cancel}")

                    trade_params = {
                        "symbol": symbol,
                        "side": decision,
                        "quantity": abs(quantity),
                        "price": current_price,
                        "order_type": order_type,
                        "target_exchange": exchange_id,
                        "reduce_only": True,
                        "metadata": {"reason": f"Settlement: ROI {roi*100:.2f}% target hit via {order_type} order"}
                    }
                    res = await self.execution_agent.execute_trade(trade_params)
                    if res and res.get("status") == "OK":
                        log.info(f"SETTLEMENT: Logging outcome for {symbol} on {exchange_id} with ROI {roi*100:.2f}%")
                        self.db.log_trade_outcome(symbol, exchange_id, roi)

        except Exception as e:
            log.error(f"Settlement Cycle Error: {e}")

    async def reconcile_all(self):
        """Sync DB positions with actual exchange state."""
        for eid in ["hyperliquid", "bingx"]:
            try:
                # Get DB positions before reconciliation to detect closed positions
                db_positions_before = self.db.get_positions()

                raw_positions = await self.execution_agent.multi_client.get_onchain_positions(eid)
                # Map CCXT positions to our internal format
                # [{'symbol': 'BTC/USDT', 'quantity': 1.0, 'avg_price': 50000}, ...]
                sync_list = []
                for p in raw_positions:
                    # CCXT returns 'contracts' or 'amount' for quantity. 
                    # We need to handle mapping from exchange symbol back to our format
                    symbol = p.get('symbol', '')
                    if eid == "hyperliquid":
                        symbol = symbol.replace("/USDC:USDC", "/USDT")
                    
                    qty = float(p.get('contracts', 0) or p.get('amount', 0))
                    if p.get('side') == 'short': qty = -abs(qty)
                    
                    sync_list.append({
                        'symbol': symbol,
                        'quantity': qty,
                        'avg_price': float(p.get('entryPrice') or p.get('avgPrice') or 0)
                    })
                
                self.db.reconcile_positions(eid, sync_list)
                log.info(f"Reconciliation successful for {eid}")

                # Check for closed positions and cancel any resting orders for them
                actual_symbols = {p['symbol'] for p in sync_list}
                
                # Fetch all open orders once per exchange for stale check
                try:
                    open_orders = await self.execution_agent.multi_client.exchanges[eid].fetch_open_orders()
                except Exception as e:
                    log.error(f"Failed to fetch open orders for stale check on {eid}: {e}")
                    open_orders = []

                for (symbol, db_eid), pos in db_positions_before.items():
                    if db_eid == eid and symbol not in actual_symbols:
                        log.info(f"Reconciliation: Position for {symbol} on {eid} has closed. Cancelling any resting orders.")
                        
                        # Log the reconciled exit outcome (liquidations / manual exits) in DB for training feedback
                        try:
                            current_price_str = self.redis.get(f"price:{eid}:{symbol}") or self.redis.get(f"price:{symbol}")
                            if current_price_str and pos.get('avg_price', 0) > 0:
                                current_price = float(current_price_str)
                                avg_price = pos['avg_price']
                                side = "LONG" if pos['quantity'] > 0 else "SHORT"
                                if side == "LONG":
                                    roi = (current_price - avg_price) / avg_price
                                else:
                                    roi = (avg_price - current_price) / avg_price
                                log.warning(f"Reconciliation: Logging reconciled exit outcome for {symbol} on {eid} with estimated ROI {roi*100:.2f}%")
                                self.db.log_trade_outcome(symbol, eid, roi)
                        except Exception as ex_outcome:
                            log.error(f"Failed to log reconciled trade outcome for {symbol} on {eid}: {ex_outcome}")

                        mapped_symbol = self.execution_agent.multi_client._get_mapped_symbol(eid, symbol)
                        for order in open_orders:
                            if order.get('symbol') == mapped_symbol:
                                try:
                                    log.info(f"Reconciliation: Cancelling stale order {order['id']} for {symbol} on {eid}")
                                    await self.execution_agent.multi_client.exchanges[eid].cancel_order(order['id'], mapped_symbol)
                                except Exception as ex:
                                    log.error(f"Failed to cancel resting orders for closed position {symbol} on {eid}: {ex}")

                # Cancel ANY open order that is older than 2 minutes (120 seconds) to prevent stuck unfilled limit orders
                import time
                current_time_ms = time.time() * 1000
                for order in open_orders:
                    order_time = order.get('timestamp')
                    if order_time and (current_time_ms - order_time) > 120000:
                        symbol = order.get('symbol')
                        order_id = order.get('id')
                        log.warning(f"[STALE_ORDER] Order {order_id} for {symbol} on {eid} is older than 2 minutes. Cancelling...")
                        try:
                            await self.execution_agent.multi_client.exchanges[eid].cancel_order(order_id, symbol)
                        except Exception as ex:
                            log.error(f"Failed to cancel stale order {order_id} on {eid}: {ex}")

            except Exception as e:
                log.error(f"Failed to reconcile {eid}: {e}")

    async def run_forever(self):
        log.info("Starting Autonomous Settlement & Profit Booking Agent...")
        import time
        last_reconcile = 0.0
        try:
            while True:
                try:
                    current_time = time.time()
                    # 1. Mandatory Sync with On-Chain Truth (slow REST queries)
                    if current_time - last_reconcile >= 60.0:
                        await self.reconcile_all()
                        last_reconcile = current_time

                    # 2. Run logic on actual state (fast Redis-based SL/TP checking)
                    await self.run_settlement_cycle("hyperliquid")
                    await self.run_settlement_cycle("bingx")
                except Exception as e:
                    log.error(f"Settlement Agent Loop Error: {e}")
                await asyncio.sleep(10) # Reconcile and check every 10 seconds
        finally:
            # Clean up client connections on shutdown only, not every cycle
            try:
                await self.execution_agent.multi_client.close()
            except Exception as close_err:
                log.error(f"Error closing multi-client in settlement: {close_err}")


