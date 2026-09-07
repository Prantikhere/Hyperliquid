import polars as pl
import pandas as pd
from datetime import datetime, timezone
from src.utils.db import DatabaseManager
from src.utils.logger import log

class FeatureEngine:
    def __init__(self, db_manager):
        self.db = db_manager

    def get_market_features(self, market_id, outcome_index=0, lookback_minutes=60):
        """Fetch raw ticks and calculate features."""
        query = f"""
        SELECT time, mid_price, bid_size, ask_size, spread
        FROM order_book_ticks
        WHERE market_id = '{market_id}' AND outcome_index = {outcome_index}
        AND time > NOW() - INTERVAL '{lookback_minutes} minutes'
        ORDER BY time ASC
        """
        data = self.db.execute_query(query)
        if not data or len(data) < 10:
            return None
        
        # Convert Postgres Decimals to float and ensure numeric types
        processed_data = []
        for row in data:
            processed_data.append({
                "time": row[0],
                "mid_price": float(row[1]),
                "bid_size": float(row[2]),
                "ask_size": float(row[3]),
                "spread": float(row[4])
            })

        df = pl.DataFrame(processed_data)
        
        # Calculate features using Polars (fast)
        df = df.with_columns([
            (pl.col("mid_price").pct_change()).alias("returns_1t"),
            (pl.col("bid_size") / (pl.col("bid_size") + pl.col("ask_size") + 1e-9)).alias("order_imbalance"),
            (pl.col("spread") / (pl.col("mid_price") + 1e-9)).alias("relative_spread")
        ])
        
        # Rolling features
        df = df.with_columns([
            pl.col("mid_price").rolling_mean(window_size=10).alias("sma_10t"),
            pl.col("mid_price").rolling_std(window_size=10).alias("vol_10t")
        ])
        
        # FINAL HARDENING: 
        # 1. Fill NAs from rolling/pct_change
        df = df.fill_nan(0).fill_null(0)
        
        # 2. Extract last row
        last_row_df = df.tail(1)
        if last_row_df.is_empty():
            return None
            
        # 3. Convert to dict and remove non-numeric 'time'
        last_row = last_row_df.to_dicts()[0]
        last_row.pop("time", None)
        
        # 4. Final check: ensure all values are floats to prevent XGBoost object error
        return {k: float(v) for k, v in last_row.items()}

    def get_external_features(self, symbol="BTC/USDT", lookback_minutes=60):
        query = f"""
        SELECT time, price
        FROM external_prices
        WHERE symbol = '{symbol}'
        AND time > NOW() - INTERVAL '{lookback_minutes} minutes'
        ORDER BY time ASC
        """
        data = self.db.execute_query(query)
        if not data:
            return None
        
        processed_data = []
        for row in data:
            processed_data.append({
                "time": row[0],
                "price": float(row[1])
            })

        df = pl.DataFrame(processed_data)
        df = df.with_columns([
            pl.col("price").pct_change().alias("ext_returns_1t")
        ])
        
        last_row = df.tail(1).to_dicts()[0] if not df.is_empty() else None
        if last_row:
            last_row.pop("time", None)
            return {k: float(v) for k, v in last_row.items()}
            
        return None
