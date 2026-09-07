import asyncio
import os
from dotenv import load_dotenv
from src.utils.logger import log
from src.agents.supervisor import SupervisorAgent

load_dotenv()

class BingXExecutor:
    def __init__(self):
        self.supervisor = SupervisorAgent()
        self.symbols = [
            "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
            "DOGE/USDT", "ADA/USDT", "TRX/USDT", "AVAX/USDT", "DOT/USDT",
            "LINK/USDT", "SHIB/USDT", "NEAR/USDT", "LTC/USDT", "UNI/USDT"
        ]
        self.interval = int(os.getenv("CHECK_INTERVAL", 300))

    async def run(self):
        log.info(f"Starting Dedicated BingX Executor (Interval: {self.interval}s)...")
        try:
            while True:
                try:
                    for symbol in self.symbols:
                        await self.supervisor.run_cycle(symbol, exchange_id="bingx")
                        await asyncio.sleep(1) # Rate limit protection

                    log.info(f"BingX scan complete. Sleeping...")
                    await asyncio.sleep(self.interval)
                except Exception as e:
                    log.error(f"BingX Executor Error: {e}")
                    await asyncio.sleep(10)
        finally:
            # Release FDs/sockets on shutdown only, not every sweep iteration
            try:
                await self.supervisor.execution_agent.multi_client.close()
            except Exception as close_err:
                log.error(f"Error closing multi-client in BingX executor: {close_err}")

if __name__ == "__main__":
    executor = BingXExecutor()
    asyncio.run(executor.run())
