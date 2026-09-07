import numpy as np
from src.execution.risk import RiskManager
from src.utils.logger import log

class RiskAgent:
    def __init__(self, db=None):
        self.risk_manager = RiskManager(db=db)

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

            # Enforce minimum notional for HL testnet
            if position_usd < 10.0:
                return {"approved": False, "reason": f"Position ${position_usd:.2f} below $10 minimum (scale={scale_factor:.2f}, sortino={sortino_multiplier:.2f})"}

            # Hard cap: never risk more than 25% of bankroll on a single position
            max_position = self.risk_manager.bankroll * 0.25
            if position_usd > max_position:
                position_usd = max_position

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
