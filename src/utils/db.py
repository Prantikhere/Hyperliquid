import os
import psycopg2
import json
from psycopg2.extras import execute_values
from datetime import datetime, timezone
from dotenv import load_dotenv
from src.utils.logger import log

load_dotenv()

class DatabaseManager:
    def __init__(self):
        self.conn_str = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST', 'timescaledb')}:{os.getenv('DB_PORT', '5432')}/{os.getenv('DB_NAME')}"
        self.conn = None
        self.connect()

    def connect(self):
        try:
            self.conn = psycopg2.connect(self.conn_str)
            self.conn.autocommit = True
            log.info("Successfully connected to Database")
        except Exception as e:
            log.error(f"Failed to connect to database: {e}")
            raise

    def ensure_connection(self):
        if self.conn is None or self.conn.closed != 0:
            log.info("Database connection is closed or None. Reconnecting...")
            try:
                self.connect()
            except Exception as e:
                log.error(f"Failed to reconnect to database: {e}")
            return

        # Double check with a quick test query
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1")
        except Exception as e:
            log.info(f"Database connection check failed ({e}). Reconnecting...")
            try:
                self.conn.close()
            except Exception:
                pass
            try:
                self.connect()
            except Exception as re_err:
                log.error(f"Failed to reconnect to database: {re_err}")

    def execute_query(self, query, params=None):
        self.ensure_connection()
        with self.conn.cursor() as cur:
            cur.execute(query, params)
            return cur.fetchall() if cur.description else None

    def insert_external_price(self, data):
        self.ensure_connection()
        query = "INSERT INTO external_prices (time, symbol, price, volume) VALUES (%s, %s, %s, %s)"
        ts = data[0] if data[0] is not None else datetime.now(timezone.utc)
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, (ts, data[1], data[2], data[3]))
        except Exception as e:
            log.error(f"Error inserting external price: {e}")

    def insert_trade(self, symbol, exchange_id, side, price, quantity, status, metadata=None):
        """Modified to support metadata for explainable decisions."""
        self.ensure_connection()
        query = """
        INSERT INTO system_trades (time, market_id, exchange_id, side, price, size, status, metadata)
        VALUES (NOW(), %s, %s, %s, %s, %s, %s, %s)
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, (symbol, exchange_id, side, price, quantity, status, json.dumps(metadata) if metadata else None))
        except Exception as e:
            log.error(f"Error inserting trade: {e}")

    def log_trade_outcome(self, symbol, exchange_id, roi):
        """Record the outcome of the most recent trade for this position for training."""
        self.ensure_connection()
        query = """
        UPDATE system_trades 
        SET metadata = jsonb_set(COALESCE(metadata, '{}'::jsonb), '{outcome}', %s::jsonb)
        WHERE id = (
            SELECT id FROM system_trades 
            WHERE market_id = %s AND exchange_id = %s AND status = 'LIVE_OK'
              AND (metadata->>'quant_signals') IS NOT NULL
            ORDER BY time DESC LIMIT 1
        )
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(query, (json.dumps(roi), symbol, exchange_id))
        except Exception as e:
            log.error(f"Error logging trade outcome: {e}")

    def update_position(self, symbol, exchange_id, price, quantity):
        self.ensure_connection()
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT quantity, avg_price FROM positions WHERE symbol = %s AND exchange_id = %s", (symbol, exchange_id))
                row = cur.fetchone()
                if row:
                    curr_qty, curr_avg = row
                    new_qty = curr_qty + quantity
                    
                    # If position is closed (flipped or zeroed)
                    if (curr_qty > 0 and new_qty <= 0) or (curr_qty < 0 and new_qty >= 0):
                        if new_qty == 0:
                            cur.execute("DELETE FROM positions WHERE symbol = %s AND exchange_id = %s", (symbol, exchange_id))
                        else:
                            # Flipped position: New quantity has new entry price
                            cur.execute("UPDATE positions SET quantity = %s, avg_price = %s, updated_at = NOW() WHERE symbol = %s AND exchange_id = %s", (new_qty, price, symbol, exchange_id))
                    else:
                        # Position increased or reduced but not closed/flipped
                        is_increase = (curr_qty > 0 and quantity > 0) or (curr_qty < 0 and quantity < 0)
                        if is_increase:
                            # Weighted average for increase
                            new_avg = ((curr_avg * abs(curr_qty)) + (price * abs(quantity))) / abs(new_qty)
                        else:
                            # Avg price stays same on reduction
                            new_avg = curr_avg
                        
                        cur.execute("UPDATE positions SET quantity = %s, avg_price = %s, updated_at = NOW() WHERE symbol = %s AND exchange_id = %s", (new_qty, new_avg, symbol, exchange_id))
                else:
                    # New position
                    if quantity != 0:
                        cur.execute("INSERT INTO positions (symbol, exchange_id, avg_price, quantity) VALUES (%s, %s, %s, %s)", (symbol, exchange_id, price, quantity))
        except Exception as e:
            log.error(f"Error updating position: {e}")

    def reconcile_positions(self, exchange_id, onchain_positions):
        """
        Force local DB to match the exchange truth (onchain_positions).
        onchain_positions is a list of dicts: [{'symbol': 'BTC/USDT', 'quantity': 1.0, 'avg_price': 50000}, ...]
        """
        self.ensure_connection()
        try:
            with self.conn.cursor() as cur:
                # 1. Get current symbols in DB for this exchange
                cur.execute("SELECT symbol FROM positions WHERE exchange_id = %s", (exchange_id,))
                db_symbols = {row[0] for row in cur.fetchall()}
                
                # 2. Update or Insert actual positions
                actual_symbols = set()
                for pos in onchain_positions:
                    symbol = pos['symbol']
                    qty = pos['quantity']
                    price = pos['avg_price']
                    actual_symbols.add(symbol)
                    
                    if qty == 0:
                        cur.execute("DELETE FROM positions WHERE symbol = %s AND exchange_id = %s", (symbol, exchange_id))
                        continue
                        
                    cur.execute("SELECT 1 FROM positions WHERE symbol = %s AND exchange_id = %s", (symbol, exchange_id))
                    if cur.fetchone():
                        cur.execute("UPDATE positions SET quantity = %s, avg_price = %s, updated_at = NOW() WHERE symbol = %s AND exchange_id = %s", (qty, price, symbol, exchange_id))
                    else:
                        cur.execute("INSERT INTO positions (symbol, exchange_id, avg_price, quantity) VALUES (%s, %s, %s, %s)", (symbol, exchange_id, price, qty))
                
                # 3. Remove positions that no longer exist on-chain
                for symbol in db_symbols:
                    if symbol not in actual_symbols:
                        log.info(f"Reconciliation: Removing defunct position {symbol} from {exchange_id}")
                        cur.execute("DELETE FROM positions WHERE symbol = %s AND exchange_id = %s", (symbol, exchange_id))
        except Exception as e:
            log.error(f"Reconciliation Error for {exchange_id}: {e}")

    def get_positions(self):
        self.ensure_connection()
        try:
            with self.conn.cursor() as cur:
                cur.execute("SELECT symbol, exchange_id, quantity, avg_price FROM positions")
                return {(row[0], row[1]): {"quantity": row[2], "avg_price": row[3]} for row in cur.fetchall()}
        except Exception as e:
            log.error(f"Error getting positions: {e}")
            return {}
