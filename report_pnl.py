import asyncio
import os
import redis
import json
from src.utils.db import DatabaseManager
from src.utils.logger import log
from src.execution.multi_client import MultiExchangeClient
from dotenv import load_dotenv

load_dotenv()

async def report_pnl():
    db = DatabaseManager()
    r = redis.Redis(host=os.getenv('REDIS_HOST', '127.0.0.1'), port=6379, decode_responses=True)
    multi_client = MultiExchangeClient()
    
    start_capital = float(os.getenv("BANKROLL", 467.06))
    
    while True:
        try:
            log.info("--- Generating EXPLAINABLE PnL Report ---")
            
            # 1. HL ON-CHAIN
            hl_balance = await multi_client.get_balance('hyperliquid')
            hl_equity = float(hl_balance.get('info', {}).get('marginSummary', {}).get('accountValue', 0))
            hl_positions = await multi_client.get_onchain_positions('hyperliquid')
            
            # 2. BINGX PAPER
            local_positions = db.get_positions()
            bingx_unrealized = 0.0
            
            # 3. Decision Lineage (Explainability)
            log.info("Recent Decision Lineage (Explainable AI):")
            recent_trades = db.execute_query("""
                SELECT market_id, side, metadata->>'regime' as reg, metadata->>'confidence' as conf 
                FROM system_trades 
                ORDER BY time DESC LIMIT 3
            """)
            if recent_trades:
                for t in recent_trades:
                    log.info(f"  {t[0]} {t[1]}: Confidence {t[3]} influenced by {t[2]} regime")
            else:
                log.info("  No recent trades for lineage.")

            log.info("--- Portfolio Stats ---")
            log.info(f"  HL On-Chain Equity: ${hl_equity:.2f} USDC")
            log.info(f"  Total Session Return: ${hl_equity - start_capital:.2f}")
            log.info("--------------------------------------------------")
            
        except Exception as e:
            log.error(f"Error in PnL reporter: {e}")
            
        await asyncio.sleep(1800)

if __name__ == "__main__":
    asyncio.run(report_pnl())
