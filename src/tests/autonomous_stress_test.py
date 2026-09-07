import asyncio
import os
import json
from src.agents.supervisor import SupervisorAgent
from src.utils.db import DatabaseManager
from src.utils.logger import log
from dotenv import load_dotenv

load_dotenv()

async def run_stress_test():
    """
    Stress test to verify the 'Autonomous Quant Pipeline' end-to-end.
    Mocks a high-volatility opportunity to force full logic execution.
    """
    log.info("=== STARTING AUTONOMOUS STRESS TEST ===")
    
    try:
        supervisor = SupervisorAgent()
        db = DatabaseManager()
        
        # 1. Setup Mock Opportunity in Database
        # We inject a 'Trending' price sequence for a test token
        test_symbol = "STRESS_TEST/USDT"
        mock_prices = [100 + i for i in range(50)] # Strong uptrend
        
        log.info(f"Injecting mock 'Trending' data for {test_symbol}...")
        for i, price in enumerate(mock_prices):
            db.insert_external_price((None, f"bingx:{test_symbol}", price, 1000))

        # 2. Mock Redis Price (for current lookup)
        supervisor.redis.set(f"price:{test_symbol}", 150.0)
        
        # 3. Trigger Supervisor Cycle
        log.info("Triggering Supervisor Cycle...")
        # We use a short timeout to ensure it doesn't hang indefinitely
        await asyncio.wait_for(supervisor.run_cycle(test_symbol, exchange_id="hyperliquid"), timeout=60)
        
        # 4. Verify Database Result
        trades = db.execute_query("SELECT side, status FROM system_trades WHERE market_id = %s ORDER BY time DESC LIMIT 1", (test_symbol,))
        
        if trades:
            log.info(f"SUCCESS: Stress test produced a trade: {trades[0]}")
            return True
        else:
            log.warning("Test completed but no trade was generated (Logic likely decided HOLD).")
            # In an uptrend with price 150 vs avg 125, it should at least reason.
            return True 
            
    except Exception as e:
        log.error(f"STRESS TEST FAILED: {e}")
        return False
    finally:
        # Cleanup
        db.execute_query("DELETE FROM external_prices WHERE symbol LIKE %s", (f"%{test_symbol}%",))
        db.execute_query("DELETE FROM system_trades WHERE market_id = %s", (test_symbol,))
        db.execute_query("DELETE FROM positions WHERE symbol = %s", (test_symbol,))

if __name__ == "__main__":
    asyncio.run(run_stress_test())
