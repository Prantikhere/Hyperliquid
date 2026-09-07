
import asyncio
import os
import pandas as pd
from src.utils.db import DatabaseManager
from src.execution.multi_client import MultiExchangeClient
from src.utils.logger import log
import redis
from dotenv import load_dotenv

load_dotenv()

async def assess_performance():
    db = DatabaseManager()
    client = MultiExchangeClient()
    r = redis.Redis(host=os.getenv('REDIS_HOST', 'localhost'), port=6379, decode_responses=True)
    
    print("\n" + "="*50)
    print("      SYSTEM PERFORMANCE & PNL AUDIT")
    print("="*50)

    # 1. On-Chain Balance & Positions
    print("\n--- [1] LIVE ON-CHAIN STATE ---")
    for eid in ["hyperliquid", "bingx"]:
        print(f"\nEXCHANGE: {eid.upper()}")
        try:
            balance = await client.get_balance(eid)
            free_usdt = balance.get('USDT', {}).get('free', 0) or balance.get('USDC', {}).get('free', 0) or 0.0
            print(f"Available Capital: ${free_usdt:.2f}")
        except Exception as e:
            print(f"  Error fetching balance: {e}")
            free_usdt = 0.0

        try:
            positions = await client.get_onchain_positions(eid)
            print(f"Active Positions: {len(positions)}")
        except Exception as e:
            print(f"  Error fetching positions: {e}")
            positions = []
        
        for p in positions:
            symbol = p.get('symbol')
            qty = float(p.get('contracts', 0) or p.get('amount', 0))
            side = p.get('side', 'long')
            if side == 'short':
                qty = -abs(qty)
            entry = float(p.get('entryPrice') or 0)
            
            # Get current price from Redis
            norm_symbol = symbol.replace("/USDC:USDC", "/USDT")
            curr_price = r.get(f"price:{eid}:{norm_symbol}") or r.get(f"price:{norm_symbol}")
            
            if curr_price:
                curr_price = float(curr_price)
                unrealized_pnl = (curr_price - entry) * qty if qty > 0 else (entry - curr_price) * abs(qty)
                roi = (unrealized_pnl / (abs(qty) * entry)) * 100 if entry > 0 else 0
                print(f"  > {symbol} ({side.upper()}): Qty {qty} | Entry ${entry:.4f} | Current ${curr_price:.4f} | uPnL: ${unrealized_pnl:.2f} ({roi:.2f}%)")

    # 2. Historical Trade Performance
    print("\n--- [2] TRADE HISTORY & WIN RATE ---")
    query = """
    SELECT market_id, exchange_id, side, price, size, metadata->>'outcome' as outcome 
    FROM system_trades 
    WHERE status = 'LIVE_OK'
    ORDER BY time DESC LIMIT 50
    """
    trades = db.execute_query(query)
    
    if not trades:
        print("No live trades found in database.")
    else:
        df = pd.DataFrame(trades, columns=['symbol', 'exchange', 'side', 'price', 'size', 'outcome'])
        df['outcome'] = pd.to_numeric(df['outcome'], errors='coerce')
        
        completed_trades = df[df['outcome'].notnull()]
        if not completed_trades.empty:
            wins = completed_trades[completed_trades['outcome'] > 0]
            losses = completed_trades[completed_trades['outcome'] <= 0]
            win_rate = (len(wins) / len(completed_trades)) * 100
            total_realized_pnl_pct = completed_trades['outcome'].sum() * 100
            
            print(f"Total Completed Trades Analyzed: {len(completed_trades)}")
            print(f"Win Rate: {win_rate:.2f}%")
            print(f"Total Realized ROI: {total_realized_pnl_pct:.2f}%")
            print(f"Average ROI per trade: {(total_realized_pnl_pct / len(completed_trades)):.2f}%")
        else:
            print("No completed (closed) trades with outcomes logged yet.")

    # 3. Decision Quality
    print("\n--- [3] AI REASONING QUALITY ---")
    query_conf = "SELECT metadata->>'meta_confidence' FROM system_trades WHERE status = 'LIVE_OK' ORDER BY time DESC LIMIT 10"
    confs = db.execute_query(query_conf)
    if confs:
        avg_conf = sum([float(c[0]) for c in confs if c[0]]) / len(confs)
        print(f"Recent Executed Confidence Average: {avg_conf:.2f}")
    else:
        print("No recent executed trades to assess confidence.")

    await client.close()

if __name__ == "__main__":
    asyncio.run(assess_performance())
