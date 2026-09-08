import numpy as np
import asyncio
from src.execution.risk import RiskManager
from src.execution.multi_client import MultiExchangeClient
from src.utils.logger import log

class RiskAgent:
    def __init__(self, db=None):
        self.risk_manager = RiskManager(db=db)
        self.max_margin_usage_pct = 80.0  # Maximum margin usage percentage

    def _check_margin_available(self):
        """Check if margin usage is below threshold. Returns (ok, free_margin, usage_pct)."""
        try:
            client = MultiExchangeClient()
            balance = asyncio.get_event_loop().run_until_complete(client.hl.fetch_balance())
            info = balance.get('info', {})
            margin = info.get('marginSummary', {})
            
            account_value = float(margin.get('accountValue', 0))
            margin_used = float(margin.get('totalMarginUsed', 0))
            free_margin = float(balance.get('free', {}).get('USDC', 0))
            
            usage_pct = (margin_used / account_value * 100) if account_value > 0 else 100.0
            
            log.info(f"[RISK_AGENT] Margin check: account=\${account_value:.2f} used=\${margin_used:.2f} free=\${free_margin:.2f} usage={usage_pct:.1f}%")
            
            return usage_pct < self.max_margin_usage_pct, free_margin, usage_pct
        except Exception as e:
            log.error(f"[RISK_AGENT] Margin check failed: {e}")
            return False, 0.0, 100.0

    def evaluate_trade(self, symbol, side, confidence, current_price, regime=None, sortino=1.0):
        """Evaluate if a trade is within risk limits and calculate position size
        using Kelly criterion with regime-aware and Sortino scaling."""
        try:
            symbol_upper = symbol.upper()
            is_major = "BTC" in symbol_upper or "ETH" in symbol_upper

            # Regime-aware leverage and scale factor
            scale_factor = 1.0
            leverage = 5.0

            if regime:
                regime_upper = regime.upper()
                if "MEAN_REVERTING" in regime_upper:
                    scale_factor = 1.0
                    if confidence >= 0.8:
                        leverage = 10.0 if is_major else 7.0
                elif "TRENDING" in regime_upper:
                    scale_factor = 0.5 if is_major else 0.25
                else:
                    scale_factor = 0.5 if is_major else 0.3
            else:
                scale_factor = 1.0 if is_major else 0.5

            if scale_factor <= 0.0:
                return {"approved": False, "reason": f"Trading halted for Altcoins in {regime or 'UNKNOWN'} regime"}

            if not self.risk_manager.check_risk_limits():
                return {"approved": False, "reason": "Global risk limits exceeded"}

            # Check margin availability before sizing
            margin_ok, free_margin, usage_pct = self._check_margin_available()
            if not margin_ok:
                return {"approved": False, "reason": f"Margin usage too high: {usage_pct:.1f}% (max: {self.max_margin_usage_pct}%)"}

            # Kelly-based position sizing
            position_usd = self.risk_manager.calculate_position_size(confidence, current_price, leverage=leverage)

            # Apply regime scale factor
            position_usd = position_usd * scale_factor

            # Apply Sortino risk-adjusted multiplier (0.75 to 2.0)
            # Floor at 0.75 to fit HL testnet margin limits while clearing $10 minimum
            sortino_multiplier = float(np.clip(sortino, 0.75, 2.0))
            position_usd = position_usd * sortino_multiplier

            if position_usd <= 0:
                return {"approved": False, "reason": f"Position size 0 (scale={scale_factor:.2f}, sortino={sortino_multiplier:.2f})"}

            # Enforce minimum notional for HL testnet AFTER all scaling
            if position_usd < 10.0:
                position_usd = 10.0

            # Hard cap: never risk more than 25% of bankroll on a single position
            max_position = self.risk_manager.bankroll * 0.25
            if position_usd > max_position:
                position_usd = max_position

            # Ensure position doesn't exceed available free margin
            if position_usd > free_margin:
                position_usd = free_margin
                log.warning(f"[RISK_AGENT] Position capped to available margin: ${position_usd:.2f}")

            quantity = position_usd / current_price

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
