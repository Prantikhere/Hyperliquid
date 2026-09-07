import os
from src.utils.logger import log

class RiskManager:
    def __init__(self, bankroll=None, db=None):
        self.db = db
        # Initial Bankroll from environment
        self.bankroll = float(os.getenv("BANKROLL", bankroll or 467.06))
        # Strict Risk Rule: Max 1-2% equity risk per trade
        self.max_risk_per_trade_pct = float(os.getenv("MAX_RISK_PER_TRADE", 0.02)) 
        self.max_leverage = float(os.getenv("MAX_LEVERAGE", 5.0))
        self.daily_loss = 0.0
        self.max_daily_loss_pct = 5.0

    def calculate_position_size(self, confidence, price, leverage=5.0):
        """
        Calculates position size enforcing the strict 1-2% equity risk rule.
        Risk is defined as the amount of capital lost if the trade hits a 1% move.
        """
        if confidence < 0.4:
            return 0
            
        # Target Risk Amount (e.g., $100 * 0.01 = $1 risk)
        # We use a sliding scale between 1% and 2% based on confidence (0.4 to 1.0)
        risk_pct = 0.01 + (self.max_risk_per_trade_pct - 0.01) * ((confidence - 0.4) / 0.6)
        risk_amount = self.bankroll * risk_pct
        
        # Position Size = Risk Amount / Stop Loss Percentage (assuming 2% stop loss for crypto)
        # For simple directional bet: Size = Bankroll * risk_pct * leverage
        position_usd = self.bankroll * risk_pct * leverage
        
        # Absolute hard cap at 25% of bankroll for any single trade
        position_usd = min(position_usd, self.bankroll * 0.25)
        
        return position_usd

    def check_risk_limits(self):
        try:
            if self.db is None:
                from src.utils.db import DatabaseManager
                self.db = DatabaseManager()
            db = self.db
            # Calculate sum of realized PnL for trades completed today since UTC midnight
            query = """
            SELECT SUM((price * size) * (metadata->>'outcome')::numeric)
            FROM system_trades
            WHERE status = 'LIVE_OK'
              AND (metadata->>'outcome') IS NOT NULL
              AND time >= CURRENT_DATE
            """
            result = db.execute_query(query)
            if result and result[0][0] is not None:
                realized_pnl = float(result[0][0])
                # daily_loss is defined as a positive number representing loss
                self.daily_loss = -realized_pnl if realized_pnl < 0 else 0.0
                log.info(f"Daily realized PnL check: ${realized_pnl:+.2f} | Current daily loss: ${self.daily_loss:.2f}")
            else:
                self.daily_loss = 0.0
        except Exception as e:
            log.error(f"Error calculating daily loss from DB: {e}")
            self.daily_loss = 0.0

        limit_amount = self.bankroll * self.max_daily_loss_pct / 100.0
        if self.daily_loss >= limit_amount:
            log.warning(f"Max daily loss limit reached: ${self.daily_loss:.2f} (Limit: ${limit_amount:.2f}). Trade rejected.")
            return False
        return True

    def update_bankroll(self, new_balance):
        self.bankroll = new_balance

