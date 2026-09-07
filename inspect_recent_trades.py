import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.utils.db import DatabaseManager
import pandas as pd

def main():
    db = DatabaseManager()
    
    # 1. Print system_trades schema (columns)
    col_query = "SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'system_trades';"
    cols = db.execute_query(col_query)
    print("=== Column Names in system_trades ===")
    for col in cols:
        print(f"{col[0]} ({col[1]})")
        
    # 2. Get last 20 trades
    trades_query = """
    SELECT id, time, market_id, exchange_id, side, price, size, status, metadata
    FROM system_trades
    ORDER BY time DESC
    LIMIT 20;
    """
    rows = db.execute_query(trades_query)
    if rows:
        df = pd.DataFrame(rows, columns=['id', 'time', 'market_id', 'exchange_id', 'side', 'price', 'size', 'status', 'metadata'])
        print("\n=== LATEST 20 TRADES ===")
        print(df.to_string())
    else:
        print("\nNo trades found.")

if __name__ == "__main__":
    main()
