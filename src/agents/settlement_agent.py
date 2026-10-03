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
    def __init__(self, owned_symbols=None, brain=None):
        self.db = DatabaseManager()
        self.execution_agent = ExecutionAgent(self.db)
        self.redis = redis.Redis(host=os.getenv('REDIS_HOST', 'localhost'), port=6379, decode_responses=True)
        self.regime_engine = RegimeEngine()
        self.strategy_ensemble = StrategyEnsemble()
        self.brain = brain  # Autonomous brain for learning
        # Restrict profit-booking to this bot's own universe. Without this filter,
        # run_settlement_cycle scans every position in the DB -- including legs
        # opened by perp_ls/pairs_arb (same wallet) -- and force-closes them on
        # its own TP/SL/reversion logic, silently untracked by those bots' own logs.
        self.owned_symbols = set(owned_symbols) if owned_symbols is not None else None
        self.config_path = "models_local/strategy_config.json"

    def _load_config(self):
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}

    async def run_settlement_cycle(self, exchange_id="hyperliquid"):
        """Scan all open positions for profit booking opportunities."""
        held_hours = 0.0
        try:
            positions = self.db.get_positions()
            
            # Load active strategy configuration for dynamic mode & scalp parameters
            cfg = self._load_config()
            trading_mode = cfg.get("trading_mode", "aggressive").lower()
            scalp_cfg = cfg.get("scalp_config", {})
            is_scalp = ("scalp" in trading_mode) or scalp_cfg.get("enabled", False)
            
            # Fetch open orders once for the exchange to avoid rate limiting and allow duplicate checks
            try:
                if exchange_id == "hyperliquid":
                    # Use SDK client for HL (ccxt async hangs)
                    open_orders_raw = self.execution_agent.multi_client.hl_sdk.get_open_orders()
                    # Normalize to ccxt-like format for compatibility
                    open_orders = []
                    for o in open_orders_raw:
                        open_orders.append({
                            'symbol': o.get('coin', '') + '/USDT',
                            'id': str(o.get('oid', '')),
                            'side': 'buy' if o.get('side') == 'B' else 'sell',
                            'amount': float(o.get('sz', 0)),
                            'price': float(o.get('limitPx', 0)),
                        })
                else:
                    open_orders = await self.execution_agent.multi_client.exchanges[exchange_id].fetch_open_orders()
            except Exception as e:
                log.error(f"Failed to fetch open orders for {exchange_id} duplicate check: {e}")
                open_orders = []

            for (symbol, eid), pos in positions.items():
                if eid != exchange_id or pos['quantity'] == 0:
                    continue
                # Exclude pairs_arb reserved spread pair (ETC/FIL) which is managed by pairs_arb_executor.
                # All other open positions (including scalps that rotated off top-20 whitelist)
                # MUST be actively managed by SettlementAgent so they never become orphaned zombies.
                PAIRS_ARB_RESERVED = {"ETC/USDT", "FIL/USDT"}
                if symbol in PAIRS_ARB_RESERVED:
                    continue


                
                # DEAD SYMBOL CHECK: Skip symbols with broken oracle/exchange feeds
                dead_key = f"dead_symbols:{exchange_id}"
                dead_symbols = self.redis.smembers(dead_key)
                if symbol in dead_symbols:
                    log.warning(f"[SETTLEMENT] {symbol} on {exchange_id} is in dead_symbols set. Skipping.")
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
                if is_scalp:
                    # SCALP TP/SL SCALING:
                    # Scalp target widened between 1.0% and 2.5% (scaled by volatility, default 1.80%)
                    base_scalp_tp = scalp_cfg.get("tp_target_pct", 0.0180)
                    tp_threshold = max(0.0100, min(0.0250, base_scalp_tp * (tp_threshold / 0.025 if tp_threshold else 1.0)))
                    # Scalp stop loss capped safely between -0.70% and -1.20% (default -0.85%)
                    base_scalp_sl = scalp_cfg.get("sl_target_pct", -0.0085)
                    sl_threshold = max(-0.0120, min(-0.0070, base_scalp_sl))
                else:
                    # Enforce minimum TP threshold of 2.5% to ensure positive expectancy over fees/slippage
                    tp_threshold = max(0.025, tp_threshold)
                    # Enforce protective stop loss ceiling between -1.5% and -2.5%
                    sl_threshold = max(-0.025, min(-0.015, sl_threshold))

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

                # BAN CHECK (exit-management exempt): bans gate NEW entries via the
                # brain (brain.get_trade_recommendation), NOT exit management. Skipping
                # settlement for banned symbols orphans open positions (FIL incident:
                # short left open because settlement skipped every cycle).
                if symbol in getattr(learning_module, 'banned_symbols', set()):
                    log.warning(f"[LEARNING] Managing open position on BANNED symbol {symbol} "
                                f"on {exchange_id} (exit-only, re-entry still blocked)")

                # Get adjusted parameters based on learnings
                adjustments = learning_module.get_adjusted_parameters(symbol, exchange_id, regime)
                tp_threshold *= adjustments["take_profit_multiplier"]
                sl_threshold *= adjustments["stop_loss_multiplier"]
                if is_scalp:
                    tp_threshold = max(0.0080, min(0.0300, tp_threshold))
                    sl_threshold = max(-0.0150, min(-0.0065, sl_threshold))

                from src.quant.backtester import REVERT_MIN_ROI
                is_trend = "TRENDING" in (regime or "").upper()

                # HARD STOP LOSS: Exit immediately if loss exceeds threshold
                hard_stop = roi <= sl_threshold
                
                # DYNAMIC STOP: Tighten in unfavorable regime
                dynamic_sl = False
                if is_scalp:
                    if roi < -0.0085:
                        dynamic_sl = roi <= max(sl_threshold, -0.0085)
                else:
                    if roi < -0.02:
                        dynamic_sl = roi <= max(sl_threshold, -0.02)
                
                # TRAILING & PEAK TRACKING
                peak_key = f"peak_roi:{exchange_id}:{symbol}"
                peak_roi = float(self.redis.get(peak_key) or 0)
                if roi > peak_roi:
                    self.redis.set(peak_key, roi, ex=14400)  # 4h TTL
                    peak_roi = roi

                # TIME-BASED EXIT: Track holding time
                import time
                position_age_key = f"position_age:{exchange_id}:{symbol}"
                position_age = float(self.redis.get(position_age_key) or 0)
                if position_age == 0:
                    self.redis.set(position_age_key, time.time(), ex=86400)
                    position_age = time.time()
                held_hours = (time.time() - position_age) / 3600

                if is_scalp:
                    # SCALP TRAILING & BREAKEVEN RULES (Zero-Mistake Profit Protection)
                    # Breakeven stop: once peak reached +0.60%, exit if it drops below +0.20%
                    # Guarantees trade clears Hyperliquid fees (~0.08% roundtrip) and locks positive net outcome.
                    be_trigger = scalp_cfg.get("breakeven_trigger_pct", 0.0060)
                    be_lock = scalp_cfg.get("breakeven_lock_pct", 0.0020)
                    breakeven_stop = (peak_roi >= be_trigger) and (roi < be_lock) and (roi > -0.003)

                    # Trailing exit: once ROI >= +1.00%, exit if it drops 15% from peak
                    trail_trigger = scalp_cfg.get("trailing_trigger_pct", 0.0100)
                    trailing_exit = (roi >= trail_trigger) and (peak_roi >= trail_trigger) and (roi < peak_roi * 0.85)

                    # High profit lock: if ROI >= 1.5% and drops 12% from peak
                    high_profit_exit = (roi >= 0.015) and (peak_roi >= 0.015) and (roi < peak_roi * 0.88)

                    # Scalp stale exit: cut stagnant trades held > 1.5 hours with roi < +0.05%
                    max_scalp_hours = scalp_cfg.get("max_hold_hours", 1.5)
                    stale_position = (held_hours > max_scalp_hours) and (roi < 0.0005)

                    # Scalp reversion exit: book profit if cleared +0.60% and mean reversion occurs
                    reverted = (not is_trend) and (roi >= 0.0060) and (held_hours > 0.15) and (
                        (side == "LONG" and comp <= 0.45) or (side == "SHORT" and comp >= 0.55)
                    )
                    momentum_fading = (roi >= 0.0060) and (comp < 0.40 if side == "LONG" else comp > 0.60)
                else:
                    # Trailing exit: only if ROI > 2.5% and drops 20% from peak (locks in solid gains)
                    trailing_exit = (roi >= 0.025) and (peak_roi >= 0.025) and (roi < peak_roi * 0.80)
                    high_profit_exit = (roi >= 0.05) and (peak_roi >= 0.05) and (roi < peak_roi * 0.85)
                    breakeven_stop = (peak_roi >= 0.025) and (roi < 0.008) and (roi > -0.005)
                    stale_position = (held_hours > 4) and (roi < -0.01)
                    reverted = (not is_trend) and (roi >= 0.020) and (held_hours > 0.5) and (
                        (side == "LONG" and comp <= 0.45) or (side == "SHORT" and comp >= 0.55)
                    )
                    momentum_fading = (roi >= 0.020) and (comp < 0.35 if side == "LONG" else comp > 0.65)

                # MARKET REGIME-BASED EXITS (only with confirmed profit)
                regime_unfavorable = False
                if side == "LONG" and "HIGH_VOL" in (regime or "").upper():
                    regime_unfavorable = True
                elif side == "SHORT" and "TRENDING" in (regime or "").upper():
                    regime_unfavorable = True

                log.info(f"[SETTLEMENT] {exchange_id} {symbol} {side} | Mode: {'SCALP' if is_scalp else 'SWING'} | ROI: {roi*100:.2f}% | Peak: {peak_roi*100:.2f}% | TP: {tp_threshold*100:.2f}% | SL: {sl_threshold*100:.2f}% | comp: {comp:.2f} | regime: {regime} | held: {held_hours:.1f}h")

                # PRIORITY ORDER: Hard stop > Dynamic stop > Profit target > High profit lock > Trailing > Breakeven > Reversion > Other exits
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
                    log.info(f"$$$ PROFIT TARGET HIT for {symbol} on {exchange_id} (ROI={roi*100:.2f}%). Booking Profit.")
                    decision = "SELL" if side == "LONG" else "BUY"
                    order_type = "MARKET"
                elif high_profit_exit:
                    log.info(f"~~~ HIGH PROFIT LOCK for {symbol} on {exchange_id} (peak={peak_roi*100:.2f}%, now={roi*100:.2f}%). Locking in strong profit.")
                    decision = "SELL" if side == "LONG" else "BUY"
                    order_type = "MARKET"
                elif trailing_exit:
                    log.info(f"~~~ TRAILING EXIT for {symbol} on {exchange_id} (peak={peak_roi*100:.2f}%, now={roi*100:.2f}%). Locking profit.")
                    decision = "SELL" if side == "LONG" else "BUY"
                    order_type = "MARKET"
                elif breakeven_stop:
                    log.info(f"~~~ {'SCALP ' if is_scalp else ''}BREAKEVEN STOP for {symbol} on {exchange_id} (peak={peak_roi*100:.2f}%, now={roi*100:.2f}%). Protecting capital above fees.")
                    decision = "SELL" if side == "LONG" else "BUY"
                    order_type = "MARKET"
                elif reverted:
                    log.info(f"~~~ {'SCALP ' if is_scalp else ''}REVERSION EXIT for {symbol} on {exchange_id} (comp={comp:.2f}, ROI={roi*100:.2f}%). Booking reversion profit.")
                    decision = "SELL" if side == "LONG" else "BUY"
                    order_type = "MARKET"
                elif regime_unfavorable and roi >= (0.0035 if is_scalp else 0.020):
                    log.info(f"~~~ REGIME EXIT for {symbol} on {exchange_id} (regime={regime}). Booking profit before regime change impact.")
                    decision = "SELL" if side == "LONG" else "BUY"
                    order_type = "MARKET"
                elif momentum_fading:
                    log.info(f"~~~ {'SCALP ' if is_scalp else ''}MOMENTUM EXIT for {symbol} on {exchange_id} (comp={comp:.2f}, ROI={roi*100:.2f}%). Booking profit before momentum dies.")
                    decision = "SELL" if side == "LONG" else "BUY"
                    order_type = "MARKET"
                elif stale_position:
                    log.info(f"~~~ {'SCALP ' if is_scalp else ''}STALE EXIT for {symbol} on {exchange_id} (held={held_hours:.1f}h, ROI={roi*100:.2f}%). Cutting stale position.")
                    decision = "SELL" if side == "LONG" else "BUY"
                    order_type = "MARKET"

                if decision:
                    # FAILED EXIT LIMITER: Stop retrying after 5 failures
                    failed_exit_key = f"failed_exits:{exchange_id}:{symbol}"
                    failed_count = int(self.redis.get(failed_exit_key) or 0)
                    if failed_count >= 5:
                        log.warning(f"[SETTLEMENT] {symbol} on {exchange_id}: {failed_count} failed exit attempts. Skipping to avoid infinite retry loop.")
                        continue
                    
                    mapped_symbol = self.execution_agent.multi_client._get_mapped_symbol(exchange_id, symbol)
                    
                    # CHURN GUARD: Record exit time and cooldown for this symbol
                    # Prevents the system from immediately re-entering the same position
                    cooldown_key = f"entry_cooldown:{exchange_id}:{symbol}"
                    self.redis.set(cooldown_key, time.time(), ex=3600)  # 1h cooldown
                    # Track exit count for churn detection
                    exit_count_key = f"exit_count:{exchange_id}:{symbol}:{int(time.time() / 86400)}"
                    current_exit_count = int(self.redis.get(exit_count_key) or 0)
                    self.redis.set(exit_count_key, current_exit_count + 1, ex=172800)  # 48h TTL
                    if current_exit_count + 1 >= 5:
                        log.warning(f"[CHURN_GUARD] {symbol} on {exchange_id}: {current_exit_count + 1} exits today. Increasing cooldown to 4h.")
                        self.redis.set(cooldown_key, time.time(), ex=14400)  # Extend to 4h
                    
                    # Instead of skipping, cancel any existing stale open orders for this symbol
                    for order in open_orders:
                        if order.get('symbol') == mapped_symbol:
                            try:
                                log.info(f"[SETTLEMENT] Canceling stale open order {order.get('id')} for {symbol} on {exchange_id}...")
                                if exchange_id == "hyperliquid":
                                    # SDK signature: cancel_order(coin, oid) -- exchanges dict is None for HL
                                    coin = mapped_symbol.replace("/USDT", "")
                                    self.execution_agent.multi_client.hl_sdk.cancel_order(coin, order.get('id'))
                                else:
                                    await self.execution_agent.multi_client.exchanges[exchange_id].cancel_order(order.get('id'), mapped_symbol)
                            except Exception as ex_cancel:
                                log.error(f"Failed to cancel order {order.get('id')} on {exchange_id}: {ex_cancel}")

                    # Use Redis-cached price from price_daemon (aiohttp, no hanging)
                    # HL's oracle uses allMids which is the same source
                    live_price = current_price

                    trade_params = {
                        "symbol": symbol,
                        "side": decision,
                        "quantity": abs(quantity),
                        "price": live_price,
                        "order_type": order_type,
                        "target_exchange": exchange_id,
                        "reduce_only": True,
                        "metadata": {"reason": f"Settlement: ROI {roi*100:.2f}% target hit via {order_type} order"}
                    }
                    res = await self.execution_agent.execute_trade(trade_params)
                    if res and res.get("status") == "OK":
                        log.info(f"SETTLEMENT: Logging outcome for {symbol} on {exchange_id} with ROI {roi*100:.2f}%")
                        self.db.log_trade_outcome(symbol, exchange_id, roi)
                        
                        # Reset failed exit counter on success
                        failed_exit_key = f"failed_exits:{exchange_id}:{symbol}"
                        self.redis.delete(failed_exit_key)
                        self.redis.delete(f"peak_roi:{exchange_id}:{symbol}")
                        self.redis.delete(f"position_age:{exchange_id}:{symbol}")
                        
                        # SESSION LOSS TRACKING: Only track REALIZED PnL on successful exits
                        import time as _time
                        session_key = f"session_realized_pnl:{int(_time.time() / 86400)}"
                        current_session_pnl = float(self.redis.get(session_key) or 0)
                        pos_value = abs(pos.get('quantity', 0)) * avg_price if avg_price > 0 else 0
                        dollar_pnl = roi * pos_value
                        new_session_pnl = current_session_pnl + dollar_pnl
                        self.redis.set(session_key, new_session_pnl, ex=172800)  # 48h TTL
                        log.info(f"[SESSION_GUARD] Realized PnL updated: ${new_session_pnl:.2f} (limit: -$5.00)")
                        
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
                        
                        # AUTONOMOUS BRAIN: Feed trade outcome for learning
                        if self.brain:
                            try:
                                # Laya entry verdict stashed at entry -> lets forensics
                                # send /feedback (ground truth) back to the decision engine.
                                laya_entry = None
                                try:
                                    raw = self.redis.get(f"laya_entry:{exchange_id}:{symbol}")
                                    laya_entry = json.loads(raw) if raw else None
                                except Exception:
                                    laya_entry = None
                                trade_data = {
                                    "symbol": symbol,
                                    "side": side,
                                    "entry_price": avg_price,
                                    "exit_price": current_price,
                                    "roi_pct": roi * 100,  # Convert to percentage
                                    "held_seconds": held_hours * 3600,
                                    "regime_at_entry": regime,
                                    "regime_at_exit": regime,
                                    "meta_confidence": comp,
                                    "quant_action": decision,
                                    "strategy": "composite",
                                    "metadata": {"laya": laya_entry} if laya_entry else {},
                                }
                                forensics = self.brain.on_trade_exit(trade_data)
                                if forensics:
                                    log.info(f"[BRAIN] Forensics: {symbol} {forensics.failure_mode} "
                                             f"severity={forensics.severity} lesson={forensics.lesson[:60]}")
                            except Exception as brain_err:
                                log.error(f"[BRAIN] Forensics failed: {brain_err}")

                        # Direct online learning feedback to Laya decision engine
                        if laya_entry and laya_entry.get("decision_id"):
                            try:
                                from src.intelligence.laya_client import get_laya_client
                                laya_c = get_laya_client()
                                laya_c.submit_feedback(
                                    decision_id=laya_entry["decision_id"],
                                    question_id="entry",
                                    ground_truth=("enter" if roi > 0 else "avoid"),
                                    reward=float(roi),
                                    target_type="choice",
                                    notes=f"Realized ROI: {roi*100:.2f}%, held {held_hours:.1f}h"
                                )
                                log.info(f"[LAYA_FEEDBACK] Ground truth feedback sent to Laya for {symbol}: reward={roi*100:.2f}%")
                            except Exception as ex_laya_fb:
                                log.debug(f"[LAYA_FEEDBACK] Feedback submission error: {ex_laya_fb}")
                    else:
                        # Increment failed exit counter with short cooldown (2m) instead of permanent abandonment
                        failed_exit_key = f"failed_exits:{exchange_id}:{symbol}"
                        new_count = int(self.redis.get(failed_exit_key) or 0) + 1
                        ttl = 120 if new_count >= 5 else 300
                        self.redis.set(failed_exit_key, new_count, ex=ttl)
                        log.warning(f"[SETTLEMENT] {symbol} on {exchange_id}: exit order failed ({new_count}/5 retries, cooling down {ttl}s)")

        except Exception as e:
            log.exception(f"Settlement Cycle Error: {e}")

    def _sync_redis_position_state(self, exchange_id, actual_symbols):
        """Rebuild Redis position counters from on-chain truth after reconcile.

        execution_agent only incr/decrs these keys on successful orders; failed
        exits, crashes, or manual closes leave them stale (OP phantom position,
        open_positions_count vs DB mismatch, position_age for closed symbols).
        """
        try:
            # Rebuild per-symbol flags for this exchange
            prefix = f"symbol_positions:{exchange_id}:"
            for key in self.redis.scan_iter(match=f"{prefix}*"):
                symbol = key[len(prefix):]
                if symbol in actual_symbols:
                    self.redis.set(key, 1)
                else:
                    self.redis.delete(key)

            # Drop position age / peak ROI for symbols no longer open
            # (so the next entry starts with a fresh hold clock and peak)
            for pattern in (f"position_age:{exchange_id}:*", f"peak_roi:{exchange_id}:*"):
                for key in self.redis.scan_iter(match=pattern):
                    symbol = key.rsplit(":", 1)[-1]
                    if symbol not in actual_symbols:
                        self.redis.delete(key)

            # Rebuild the global open-position count from ALL exchanges' flags
            total = 0
            for key in self.redis.scan_iter(match="symbol_positions:*"):
                try:
                    total += int(self.redis.get(key) or 0)
                except (TypeError, ValueError):
                    pass
            self.redis.set("open_positions_count", max(0, total))
        except Exception as e:
            log.error(f"Redis position-state sync failed for {exchange_id}: {e}")

    async def reconcile_all(self):
        """Sync DB positions with actual exchange state.

        Hyperliquid only: BingX API key is invalid (timestamp error 109400)
        and there are no BingX positions -- reconciling it just spams errors.
        """
        for eid in ["hyperliquid"]:
            try:
                # Get DB positions before reconciliation to detect closed positions
                db_positions_before = self.db.get_positions()

                if eid == "hyperliquid":
                    raw_positions = await asyncio.wait_for(
                        self.execution_agent.multi_client.get_onchain_positions(eid),
                        timeout=15
                    )
                else:
                    raw_positions = await self.execution_agent.multi_client.get_onchain_positions(eid)
                # Map positions to our internal format
                sync_list = []
                for p in raw_positions:
                    if eid == "hyperliquid":
                        # SDK returns: {"position": {"coin": "NEAR", "szi": "-3.2", "entryPx": "4.4271", ...}}
                        pos_data = p.get("position", p)
                        coin = pos_data.get("coin", "")
                        symbol = f"{coin}/USDT"
                        qty = float(pos_data.get("szi", 0))
                        avg_price = float(pos_data.get("entryPx", 0))
                    else:
                        # CCXT format
                        symbol = p.get('symbol', '')
                        qty = float(p.get('contracts', 0) or p.get('amount', 0))
                        if p.get('side') == 'short': qty = -abs(qty)
                        avg_price = float(p.get('entryPrice') or p.get('avgPrice') or 0)
                    
                    sync_list.append({
                        'symbol': symbol,
                        'quantity': qty,
                        'avg_price': avg_price
                    })
                
                self.db.reconcile_positions(eid, sync_list)
                log.info(f"Reconciliation successful for {eid}")

                # Check for closed positions and cancel any resting orders for them
                actual_symbols = {p['symbol'] for p in sync_list}

                # Keep Redis counters aligned with on-chain truth
                self._sync_redis_position_state(eid, actual_symbols)
                
                # Fetch all open orders once per exchange for stale check
                try:
                    if eid == "hyperliquid":
                        # Use SDK client for HL
                        open_orders_raw = self.execution_agent.multi_client.hl_sdk.get_open_orders()
                        open_orders = [{'symbol': o.get('coin', '') + '/USDT', 'id': str(o.get('oid', ''))} for o in open_orders_raw]
                    else:
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
                                
                                # AUTONOMOUS BRAIN: Feed reconciled trade for learning
                                if self.brain:
                                    try:
                                        laya_entry = None
                                        try:
                                            raw = self.redis.get(f"laya_entry:{eid}:{symbol}")
                                            laya_entry = json.loads(raw) if raw else None
                                        except Exception:
                                            laya_entry = None
                                        trade_data = {
                                            "symbol": symbol,
                                            "side": side,
                                            "entry_price": avg_price,
                                            "exit_price": current_price,
                                            "roi_pct": roi * 100,
                                            "held_seconds": 0,  # Unknown for reconciliation
                                            "regime_at_entry": "UNKNOWN",
                                            "regime_at_exit": "UNKNOWN",
                                            "meta_confidence": 0.5,
                                            "quant_action": "HOLD",
                                            "strategy": "reconciliation",
                                            "metadata": {"laya": laya_entry} if laya_entry else {},
                                        }
                                        self.brain.on_trade_exit(trade_data)
                                    except Exception as brain_err:
                                        log.error(f"[BRAIN] Reconciliation forensics failed: {brain_err}")
                        except Exception as ex_outcome:
                            log.error(f"Failed to log reconciled trade outcome for {symbol} on {eid}: {ex_outcome}")

                        mapped_symbol = self.execution_agent.multi_client._get_mapped_symbol(eid, symbol)
                        for order in open_orders:
                            if order.get('symbol') == mapped_symbol:
                                try:
                                    log.info(f"Reconciliation: Cancelling stale order {order['id']} for {symbol} on {eid}")
                                    if eid == "hyperliquid":
                                        coin = mapped_symbol.replace("/USDT", "")
                                        self.execution_agent.multi_client.hl_sdk.cancel_order(coin, order['id'])
                                    else:
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
                            if eid == "hyperliquid":
                                coin = (symbol or "").replace("/USDT", "")
                                self.execution_agent.multi_client.hl_sdk.cancel_order(coin, order_id)
                            else:
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
                    # BingX disabled — API key invalid, no positions on BingX
                    # await self.run_settlement_cycle("bingx")
                except Exception as e:
                    log.error(f"Settlement Agent Loop Error: {e}")
                await asyncio.sleep(10) # Reconcile and check every 10 seconds
        finally:
            # Clean up client connections on shutdown only, not every cycle
            try:
                await self.execution_agent.multi_client.close()
            except Exception as close_err:
                log.error(f"Error closing multi-client in settlement: {close_err}")


