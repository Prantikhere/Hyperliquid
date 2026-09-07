import numpy as np
from src.execution.risk import RiskManager
from src.utils.logger import log

class RiskAgent:
    def __init__(self, db=None):
        self.risk_manager = RiskManager(db=db)

    def evaluate_trade(self, symbol, side, confidence, current_price, regime=None, sortino=1.0):
        """Evaluate if a trade is within risk limits and calculate position size with regime-aware and Sortino scaling."""
        try:
            # 1. Determine Asset Class (Majors: BTC/ETH vs Altcoins)
            symbol_upper = symbol.upper()
            is_major = "BTC" in symbol_upper or "ETH" in symbol_upper
            
            # 2. Determine dynamic leverage and regime-aware scaling factor
            scale_factor = 1.0
            leverage = 5.0
            
            if regime:
                regime_upper = regime.upper()
                if "MEAN_REVERTING" in regime_upper:
                    # Validated primary edge (with or without HIGH_VOL): full sizing.
                    scale_factor = 1.0
                    if confidence >= 0.8:
                        leverage = 10.0 if is_major else 7.0
                elif "TRENDING" in regime_upper:
                    # Trend is not a validated edge: majors half, altcoins quarter.
                    scale_factor = 0.5 if is_major else 0.25
                else:
                    # Neutral: majors half, altcoins scaled down to 20%.
                    scale_factor = 0.5 if is_major else 0.2
            else:
                # Fallback if no regime provided: Majors 1.0, Altcoins 0.5
                scale_factor = 1.0 if is_major else 0.5
                
            if scale_factor <= 0.0:
                return {"approved": False, "reason": f"Trading halted for Altcoins in {regime or 'UNKNOWN'} regime"}

            # 3. Check global risk limits
            if not self.risk_manager.check_risk_limits():
                return {"approved": False, "reason": "Global risk limits exceeded"}

            # 4. Calculate position size
            model_prob = confidence 
            position_usd = self.risk_manager.calculate_position_size(model_prob, current_price, leverage=leverage)
            
            # Apply scaling factor
            position_usd = position_usd * scale_factor
            
            # Apply Sortino risk-adjusted multiplier (0.5 to 1.5) — floor raised from 0.25 so
            # low-sortino paths still produce actionable position sizes above exchange minimums.
            sortino_multiplier = float(np.clip(sortino, 0.5, 1.5))
            position_usd = position_usd * sortino_multiplier
            
            if position_usd <= 0:
                return {"approved": False, "reason": f"Position size calculation resulted in 0 or negative (scale factor: {scale_factor:.2f}, sortino mult: {sortino_multiplier:.2f})"}
            
            # Enforce minimum notional for HL testnet (~$10) — reject dust orders
            if position_usd < 10.0:
                return {"approved": False, "reason": f"Position size ${position_usd:.2f} below minimum $10 notional"}
            
            quantity = position_usd / current_price
            
            log.info(f"[RISK_AGENT] Sizing evaluation: {symbol} | Regime: {regime} | Scale Factor: {scale_factor:.2f} | Sortino Mult: {sortino_multiplier:.2f} | Leverage: {leverage}x | Final Position: ${position_usd:.2f}")
            
            return {
                "approved": True,
                "position_usd": position_usd,
                "quantity": quantity,
                "symbol": symbol,
                "side": side,
                "price": current_price, # Include price for execution and logging
                "leverage": leverage
            }
        except Exception as e:
            log.error(f"RiskAgent Error: {e}")
            return {"approved": False, "reason": str(e)}
