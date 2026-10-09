import os
import numpy as np
from src.execution.risk import RiskManager
from src.execution.multi_client import MultiExchangeClient
from src.utils.logger import log

class RiskAgent:
    def __init__(self, db=None):
        self.risk_manager = RiskManager(db=db)
        self.max_margin_usage_pct = 80.0  # Maximum margin usage percentage
        self.max_concurrent_positions = 5  # Base maximum open positions across all symbols
        self.max_concurrent_positions_aggressive = 7  # Expanded when circumstances permit
        self.max_drawdown_pct = 15.0  # Maximum drawdown from peak before kill switch
        self.max_single_loss_pct = 3.0  # Maximum loss per trade (3%)
        self.min_risk_reward = 1.5  # Minimum risk/reward ratio (1:1.5)
        self.max_positions_per_symbol = 1  # Maximum 1 position per symbol (no averaging)
        self.max_daily_trades_per_symbol = 6  # Maximum 6 trades per symbol per day

    def _check_margin_available(self):
        """Check if margin usage is below threshold. Returns (ok, free_margin, usage_pct, account_value)."""
        try:
            import ccxt
            import os
            from dotenv import load_dotenv
            load_dotenv()
            
            hl_key = os.getenv("HL_PRIVATE_KEY", "")
            primary_addr = os.getenv("HL_WALLET_ADDRESS", "")
            
            exchange = ccxt.hyperliquid({
                'privateKey': hl_key,
                'walletAddress': primary_addr,
                'enableRateLimit': True,
                'options': {'defaultType': 'swap'},
                'timeout': 30000,
            })
            exchange.set_sandbox_mode(True)
            exchange.walletAddress = primary_addr

            balance = exchange.fetch_balance()

            info = balance.get('info', {})
            margin = info.get('marginSummary', {})

            account_value = float(margin.get('accountValue', 0))
            margin_used = float(margin.get('totalMarginUsed', 0))
            free_margin = float(balance.get('free', {}).get('USDC', 0))

            usage_pct = (margin_used / account_value * 100) if account_value > 0 else 100.0

            log.info(f"[RISK_AGENT] Margin check: account=${account_value:.2f} used=${margin_used:.2f} free=${free_margin:.2f} usage={usage_pct:.1f}%")

            return usage_pct < self.max_margin_usage_pct, free_margin, usage_pct, account_value
        except Exception as e:
            log.error(f"[RISK_AGENT] Margin check failed: {e}")
            return False, 0.0, 100.0, 0.0

    def evaluate_trade(self, symbol, side, confidence, current_price, regime=None, sortino=1.0, exchange_id='hyperliquid'):
        """Evaluate if a trade is within risk limits and calculate position size
        using Kelly criterion with regime-aware and Sortino scaling."""
        try:
            symbol_upper = symbol.upper()
            is_major = "BTC" in symbol_upper or "ETH" in symbol_upper

            if not self.risk_manager.check_risk_limits():
                return {"approved": False, "reason": "Global risk limits exceeded"}

            # Check margin availability before sizing
            margin_ok, free_margin, usage_pct, account_value = self._check_margin_available()
            if not margin_ok:
                return {"approved": False, "reason": f"Margin usage too high: {usage_pct:.1f}% (max: {self.max_margin_usage_pct}%)"}

            # Circumstances permit margin check:
            # - Free margin ample (>= $45.00)
            # - Current margin usage modest (< 50.0%)
            circumstances_permit_margin = (free_margin >= 45.0 and usage_pct < 50.0)

            # MAX DRAWDOWN KILL SWITCH: Halt trading if drawdown exceeds threshold
            try:
                import redis as _redis
                _r = _redis.Redis(host=os.getenv('REDIS_HOST', 'localhost'), port=6379, decode_responses=True)
                peak_key = "account_peak_value"
                current_key = "account_current_value"
                peak_value = float(_r.get(peak_key) or 0)
                current_value = float(_r.get(current_key) or 0)
                if peak_value > 0 and current_value > 0:
                    drawdown_pct = ((peak_value - current_value) / peak_value) * 100
                    if drawdown_pct >= self.max_drawdown_pct:
                        return {"approved": False, "reason": f"MAX DRAWDOWN KILL SWITCH: {drawdown_pct:.1f}% drawdown (limit: {self.max_drawdown_pct}%)"}
            except Exception:
                pass  # Redis unavailable, skip check

            # CONCURRENT POSITION GUARD: Limit number of open positions to prevent overexposure
            max_positions = 4 if circumstances_permit_margin else 3
            try:
                import redis as _redis
                _r = _redis.Redis(host=os.getenv('REDIS_HOST', 'localhost'), port=6379, decode_responses=True)
                positions_key = "open_positions_count"
                open_count = int(_r.get(positions_key) or 0)
                if open_count >= max_positions:
                    return {"approved": False, "reason": f"Too many open positions: {open_count} (max: {max_positions})"}

                # DIRECTIONAL CORRELATION GUARD: Cap correlated crypto exposure to avoid basket dumps
                if self.risk_manager and hasattr(self.risk_manager, "db") and self.risk_manager.db:
                    db_positions = self.risk_manager.db.get_positions()
                    crypto_same_side = sum(
                        1 for (sym, eid), p in db_positions.items()
                        if eid == exchange_id and abs(float(p.get("quantity", 0) or 0)) > 1e-6
                        and "PAXG" not in sym
                        and ((side.upper() == "BUY" and float(p.get("quantity", 0) or 0) > 0) or
                             (side.upper() == "SELL" and float(p.get("quantity", 0) or 0) < 0))
                    )
                    if crypto_same_side >= 2:
                        return {"approved": False, "reason": f"Directional correlation guard: already {crypto_same_side} crypto {side}s open"}
            except Exception as e:
                log.debug(f"[RISK_AGENT] Position count/correlation check error: {e}")

            # Regime-aware leverage and scale factor
            scale_factor = 1.0
            leverage = 5.0

            if regime:
                regime_upper = regime.upper()
                if "MEAN_REVERTING" in regime_upper:
                    scale_factor = 1.25 if (circumstances_permit_margin and confidence >= 0.55) else 1.0
                    if confidence >= 0.70 and circumstances_permit_margin:
                        leverage = 10.0 if is_major else 7.0
                    elif confidence >= 0.8:
                        leverage = 10.0 if is_major else 7.0
                elif "TRENDING" in regime_upper:
                    if circumstances_permit_margin and confidence >= 0.55:
                        scale_factor = 0.8 if is_major else 0.5
                        leverage = 7.0 if is_major else 5.0
                    else:
                        scale_factor = 0.5 if is_major else 0.25
                else:
                    # NEUTRAL regime
                    if circumstances_permit_margin and confidence >= 0.55:
                        scale_factor = 1.1 if is_major else 0.85
                    else:
                        scale_factor = 0.8 if is_major else 0.6
            else:
                scale_factor = 1.2 if (circumstances_permit_margin and is_major) else (1.0 if is_major else 0.5)

            if scale_factor <= 0.0:
                return {"approved": False, "reason": f"Trading halted for Altcoins in {regime or 'UNKNOWN'} regime"}

            # SYMBOL DIVERSIFICATION GUARD: Limit positions per symbol and daily trades per symbol
            try:
                import redis as _redis
                import time as _time
                _r = _redis.Redis(host=os.getenv('REDIS_HOST', 'localhost'), port=6379, decode_responses=True)
                
                # Check if already have a position in this symbol
                symbol_positions_key = f"symbol_positions:{exchange_id}:{symbol}"
                symbol_positions = int(_r.get(symbol_positions_key) or 0)
                log.debug(f"[RISK_DEBUG] {symbol}: symbol_positions={symbol_positions}, max={self.max_positions_per_symbol}")
                if symbol_positions >= self.max_positions_per_symbol:
                    return {"approved": False, "reason": f"Already have {symbol_positions} position(s) in {symbol} (max: {self.max_positions_per_symbol})"}
                
                # Check daily trade count for this symbol
                daily_trades_key = f"daily_trades:{exchange_id}:{symbol}:{int(_time.time() / 86400)}"
                daily_trades = int(_r.get(daily_trades_key) or 0)
                if daily_trades >= self.max_daily_trades_per_symbol:
                    return {"approved": False, "reason": f"Too many daily trades for {symbol}: {daily_trades} (max: {self.max_daily_trades_per_symbol})"}
            except Exception as e:
                log.warning(f"[RISK_DEBUG] Redis position check failed: {e}. Allowing trade (unsafe)")
            
            # SAFETY NET: Also check actual exchange positions directly
            try:
                from src.execution.hl_raw import HlSdkClient
                hl_client = HlSdkClient()
                coin = symbol.replace("/USDT", "")
                pos = hl_client.get_position(coin)
                if pos and float(pos.get("szi", 0)) != 0:
                    pos_side = "long" if float(pos["szi"]) > 0 else "short"
                    log.warning(f"[RISK_DEBUG] SAFETY NET: {symbol} already has active position on HL: {pos_side} {pos.get('szi')}")
                    return {"approved": False, "reason": f"Active HL position exists for {symbol}: {pos_side}"}
            except Exception as e:
                log.debug(f"[RISK_DEBUG] HL position check failed: {e}")

            # ANTI-WHIPSAW SAFETY: Reject symbols in post-stop-loss cooldown to prevent re-entering cascade
            try:
                import redis as _redis
                _r = _redis.Redis(host=os.getenv('REDIS_HOST', 'localhost'), port=6379, decode_responses=True)
                cooldown_key = f"stop_loss_cooldown:{exchange_id}:{symbol}"
                if _r.get(cooldown_key):
                    ttl_left = _r.ttl(cooldown_key)
                    log.warning(f"[RISK_AGENT] {symbol} is in post-stop-loss cooldown ({ttl_left}s remaining). Trade rejected.")
                    return {"approved": False, "reason": f"Post-stop-loss cooldown active ({ttl_left}s)"}
            except Exception:
                pass

            # Kelly-based position sizing
            position_usd = self.risk_manager.calculate_position_size(confidence, current_price, leverage=leverage)
            log.debug(f"[RISK_DEBUG] {symbol}: kelly_position=${position_usd:.2f}, confidence={confidence:.2f}, leverage={leverage:.1f}")

            # Apply regime scale factor
            position_usd = position_usd * scale_factor
            log.debug(f"[RISK_DEBUG] {symbol}: after_scale=${position_usd:.2f}, scale_factor={scale_factor:.2f}")

            # SESSION LOSS THROTTLING: Dynamically throttle sizing if daily drawdown accumulates
            session_loss_multiplier = 1.0
            import time as _time
            import redis
            try:
                _r = redis.Redis(host=os.getenv('REDIS_HOST', 'localhost'), port=6379, decode_responses=True)
                session_key = f"session_realized_pnl:{int(_time.time() / 86400)}"
                session_loss = float(_r.get(session_key) or 0)
                if session_loss < -1.50:
                    session_loss_multiplier = max(0.60, 1.0 + (session_loss / 8.0))
                    log.warning(f"[SESSION_GUARD] Daily PnL ${session_loss:.2f}. Throttling notional by {(1.0-session_loss_multiplier)*100:.0f}%.")
            except Exception:
                pass

            # Scalp Sizing (Sized for full capital deployment with active drawdown throttling):
            # Normal: Target notional $90 - $165 depending on free margin & confidence
            # In Drawdown: Automatically throttled down to $65 - $110 to stop account bleed
            if free_margin >= 100.0:
                base_target = 150.0
            elif free_margin >= 50.0:
                base_target = 125.0
            else:
                base_target = 95.0

            if confidence >= 0.55:
                base_target += 15.0
            elif confidence < 0.48:
                base_target -= 15.0

            target_notional = max(60.0, min(165.0, base_target * session_loss_multiplier))
            position_usd = target_notional

            # Ensure margin required doesn't exceed 40% of free margin
            req_margin = position_usd / max(1.0, leverage)
            if req_margin > free_margin * 0.40:
                position_usd = free_margin * 0.40 * leverage
                log.info(f"[RISK_AGENT] Notional clamped to 40% free margin: ${position_usd:.2f}")

            # Enforce minimum notional for HL ($10 minimum, using $12 floor)
            if position_usd < 12.0:
                position_usd = 12.0

            # Absolute hard cap on notional: $165.0
            position_usd = min(165.0, position_usd)



            quantity = position_usd / current_price
            sortino_multiplier = max(0.8, min(1.5, sortino or 1.0))

            log.info(f"[RISK_AGENT] Kelly sizing: {symbol} | conf={confidence:.2f} | regime={regime} "
                     f"scale={scale_factor:.2f} sortino={sortino_multiplier:.2f} lev={leverage}x "
                     f"position=${position_usd:.2f}")

            return {
                "approved": True,
                "position_usd": position_usd,
                "quantity": quantity,
                "symbol": symbol,
                "side": side,
                "price": current_price,
                "leverage": leverage
            }
        except Exception as e:
            log.error(f"RiskAgent Error: {e}")
            return {"approved": False, "reason": str(e)}
