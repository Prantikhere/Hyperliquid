import asyncio
import os
import json
from src.utils.logger import log
from src.utils.db import DatabaseManager
from src.agents.execution_agent import ExecutionAgent
from src.quant.regime_engine import RegimeEngine
from src.quant.multi_strategy import StrategyEnsemble
from src.quant.learning_module import learning_module
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

                # LEARNING CHECK: Check if we should trade this symbol based on past failures
                should_trade = learning_module.should_trade_symbol(symbol, exchange_id)
                if not should_trade:
                    log.warning(f"[LEARNING] Skipping {symbol} on {exchange_id}: Too many past failures")
                    continue

                # Get adjusted parameters based on learnings
                adjustments = learning_module.get_adjusted_parameters(symbol, exchange_id, regime)
                tp_threshold *= adjustments["take_profit_multiplier"]
                sl_threshold *= adjustments["stop_loss_multiplier"]

                # Trend-following positions ride to TP/SL; only mean-reversion / neutral trades
                # book at reversion-to-mean. Mirrors backtester.simulate exit model.
                # Reversion exit must clear round-trip fees with margin (mirrors backtester
                # REVERT_MIN_ROI). Booking a reversion below ~0.15% ROI is a net loss after fees.
                from src.quant.backtester import REVERT_MIN_ROI
                is_trend = "TRENDING" in (regime or "").upper()

                # HARD STOP LOSS: Exit immediately if loss exceeds threshold
                # This is the primary defense against large losses
                hard_stop = roi <= sl_threshold
                
                # DYNAMIC STOP: Tighten stops as loss increases
                # If loss > 1%, tighten stop to 1.5%
                # If loss > 2%, tighten stop to 2%
                # This prevents losses from growing beyond controlled levels
                dynamic_sl = False
                if roi < -0.01:  # Loss > 1%
                    dynamic_sl = roi <= max(sl_threshold, -0.015)  # Tighten to 1.5%
                if roi < -0.02:  # Loss > 2%
                    dynamic_sl = roi <= max(sl_threshold, -0.02)  # Tighten to 2%
                
                # TRAILING STOP: Lock in gains as price moves in our favor
                peak_key = f"peak_roi:{exchange_id}:{symbol}"
                peak_roi = float(self.redis.get(peak_key) or 0)
                if roi > peak_roi:
                    self.redis.set(peak_key, roi, ex=14400)  # 4h TTL
                    peak_roi = roi
                
                # Aggressive trailing: exit if ROI drops 15% from peak (was 20%)
                trailing_exit = (roi > 0.003) and (peak_roi > 0.003) and (roi < peak_roi * 0.85)
                
                # High profit lock: exit if ROI > 4% and drops 10% from peak
                high_profit_exit = (roi > 0.04) and (peak_roi > 0.04) and (roi < peak_roi * 0.9)
                
                # BREAKEVEN STOP: Move stop to breakeven when ROI > 2%
                breakeven_stop = (roi > 0.02) and (peak_roi > 0.02) and (roi < 0.005)  # Near breakeven

                reverted = (not is_trend) and (roi > REVERT_MIN_ROI) and (
                    (side == "LONG" and comp <= 0.5) or (side == "SHORT" and comp >= 0.5)
                )

                # MARKET REGIME-BASED EXITS
                regime_unfavorable = False
                if side == "LONG" and "HIGH_VOL" in (regime or "").upper():
                    regime_unfavorable = True
                elif side == "SHORT" and "TRENDING" in (regime or "").upper():
                    regime_unfavorable = True

                # MOMENTUM EXIT: Exit when signal weakens
                momentum_fading = (roi > 0.01) and (comp < 0.35)

                # TIME-BASED EXIT: Cut stale positions
                import time
                position_age_key = f"position_age:{exchange_id}:{symbol}"
                position_age = float(self.redis.get(position_age_key) or 0)
                if position_age == 0:
                    self.redis.set(position_age_key, time.time(), ex=86400)
                    position_age = time.time()
                held_hours = (time.time() - position_age) / 3600
                stale_position = (held_hours > 3) and (roi < 0.03)  # 3h with <3% = stale

                log.info(f"[SETTLEMENT] {exchange_id} {symbol} {side} | ROI: {roi*100:.2f}% | Peak: {peak_roi*100:.2f}% | TP: {tp_threshold*100:.2f}% | SL: {sl_threshold*100:.2f}% | comp: {comp:.2f} | regime: {regime} | held: {held_hours:.1f}h")

                # PRIORITY ORDER: Hard stop > Dynamic stop > Profit locks > Trailing > Other exits
                decision = None
                order_type = "MARKET"
                if hard_stop:
                    log.info(f"!!! HARD STOP LOSS for {symbol} on {exchange_id} (ROI={roi*100:.2f}%). Cutting loss IMMEDIATELY.")
                    decision = "SELL" if side == "LONG" else "BUY"
                    order_type = "MARKET"
                elif dynamic_sl:
                    log.info(f"!!! DYNAMIC STOP for {symbol} on {exchange_id} (ROI={roi*100:.2f}%). Tightening stop.")
                    decision = "SELL" if side == "LONG" else "BUY"
                    order_type = "MARKET"
                elif roi >= tp_threshold:
                    log.info(f"$$$ PROFIT TARGET HIT for {symbol} on {exchange_id}. Booking Profit.")
                    decision = "SELL" if side == "LONG" else "BUY"
                    order_type = "MARKET"
                elif high_profit_exit:
                    log.info(f"~~~ HIGH PROFIT LOCK for {symbol} on {exchange_id} (peak={peak_roi*100:.2f}%, now={roi*100:.2f}%). Locking in strong profit.")
                    decision = "SELL" if side == "LONG" else "BUY"
                    order_type = "MARKET"
                elif breakeven_stop:
                    log.info(f"~~~ BREAKEVEN STOP for {symbol} on {exchange_id} (peak={peak_roi*100:.2f}%, now={roi*100:.2f}%). Protecting capital.")
                    decision = "SELL" if side == "LONG" else "BUY"
                    order_type = "MARKET"
                elif trailing_exit:
                    log.info(f"~~~ TRAILING EXIT for {symbol} on {exchange_id} (peak={peak_roi*100:.2f}%, now={roi*100:.2f}%). Locking profit.")
                    decision = "SELL" if side == "LONG" else "BUY"
                    order_type = "MARKET"
                elif reverted and roi > 0:
                    log.info(f"~~~ REVERSION EXIT for {symbol} on {exchange_id} (comp={comp:.2f}). Booking reversion profit.")
                    decision = "SELL" if side == "LONG" else "BUY"
                    order_type = "MARKET"
                elif regime_unfavorable and roi > 0:
                    log.info(f"~~~ REGIME EXIT for {symbol} on {exchange_id} (regime={regime}). Booking profit before regime change impact.")
                    decision = "SELL" if side == "LONG" else "BUY"
                    order_type = "MARKET"
                elif momentum_fading:
                    log.info(f"~~~ MOMENTUM EXIT for {symbol} on {exchange_id} (comp={comp:.2f}). Booking profit before momentum dies.")
                    decision = "SELL" if side == "LONG" else "BUY"
                    order_type = "MARKET"
                elif stale_position:
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
                        
                        # Record in learning module for future reference
                        learning_module.record_trade_outcome(
                            symbol=symbol,
                            exchange=exchange_id,
                            side=side,
                            entry_price=avg_price,
                            exit_price=current_price,
                            roi=roi,
                            regime=regime,
                            comp_score=comp,
                            holding_time_hours=held_hours
                        )

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


