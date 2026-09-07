import asyncio
import os
from dotenv import load_dotenv
from src.utils.logger import log
from src.agents.supervisor import SupervisorAgent

load_dotenv()

class ExecutionEngine:
    def __init__(self):
        self.supervisor = SupervisorAgent()
        self.symbols = [
            "BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT", "ADA/USDT",
            "BNB/USDT", "DOGE/USDT", "TRX/USDT", "AVAX/USDT", "DOT/USDT",
            "LINK/USDT", "SHIB/USDT", "NEAR/USDT", "LTC/USDT", "UNI/USDT",
            "MATIC/USDT", "ATOM/USDT", "FIL/USDT", "RENDER/USDT", "HBAR/USDT"
        ]
        self.interval = int(os.getenv("CHECK_INTERVAL", 300)) 

    async def run(self):
        mode = os.getenv('TRADING_MODE', 'paper')
        log.info(f"Starting Advanced Execution Engine (Mode: {mode}, Interval: {self.interval}s)...")
        
        while True:
            try:
                batch_size = 5
                for i in range(0, len(self.symbols), batch_size):
                    batch = self.symbols[i : i + batch_size]
                    # Note: We await them one by one or in small batches to ensure logs are readable
                    for symbol in batch:
                        try:
                            await self.supervisor.run_cycle(symbol)
                        except Exception as e:
                            log.error(f"Error in cycle for {symbol}: {e}")
                    
                    await asyncio.sleep(2)
                
                log.info(f"Full scan complete. Sleeping for {self.interval}s...")
                await asyncio.sleep(self.interval)
            except Exception as e:
                log.error(f"Error in main execution loop: {e}")
                await asyncio.sleep(10)

if __name__ == "__main__":
    execution = ExecutionEngine()
    asyncio.run(execution.run())
