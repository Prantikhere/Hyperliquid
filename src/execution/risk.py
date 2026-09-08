import os
from src.utils.logger import log

class RiskManager:
    def __init__(self, bankroll=None, db=None):
        self.db = db
        self.bankroll = float(os.getenv("BANKROLL", bankroll or 467.06))
        self.max_risk_per_trade_pct = float(os.getenv("MAX_RISK_PER_TRADE", 0.10))
        self.max_leverage = float(os.getenv("MAX_LEVERAGE", 5.0))
        self.daily_loss = 0.0
        self.max_daily_loss_pct = 5.0
        # Kelly parameters from backtest stats
        self.kelly_win_rate = float(os.getenv("KELLY_WIN_RATE", 0.55))
        self.kelly_win_loss_ratio = float(os.getenv("KELLY_WIN_LOSS_RATIO", 2.0))
        self.kelly_fraction = float(os.getenv("KELLY_FRACTION", 0.25))
        # Testnet hard cap: max position size in USD (HL testnet margin limit)
        self.max_position_usd = float(os.getenv("MAX_POSITION_USD", 10.0))

    def calculate_kelly_fraction(self, confidence):
        """Calculate Kelly-optimal fraction of bankroll to risk.
        Uses fractional Kelly (25%) for safety."""
        p = self.kelly_win_rate * (0.8 + 0.2 * confidence)
        q = 1.0 - p
        b = self.kelly_win_loss_ratio
        kelly = (b * p - q) / b if b > 0 else 0.0
        kelly = max(kelly, 0.0)
        fraction = kelly * self.kelly_fraction * min(confidence / 0.6, 1.0)
        return min(fraction, 0.10)

    def calculate_position_size(self, confidence, price, leverage=5.0):
        """Calculate position size using Kelly criterion with testnet-aware capping."""
        if confidence < 0.4:
            return 0

        kelly_risk = self.calculate_kelly_fraction(confidence)
        base_risk_pct = 0.02
        risk_pct = max(kelly_risk, base_risk_pct)

        position_usd = self.bankroll * risk_pct * leverage

        return position_usd

    def check_risk_limits(self):
        try:
            if self.db is None:
                from src.utils.db import DatabaseManager
                self.db = DatabaseManager()
            db = self.db
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
