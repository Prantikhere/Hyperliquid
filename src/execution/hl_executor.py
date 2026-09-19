import asyncio
import os
from dotenv import load_dotenv
from src.utils.logger import log
from src.agents.supervisor import SupervisorAgent
from src.agents.settlement_agent import SettlementAgent

load_dotenv()

async def run_parallel():
    print("Init Supervisor...")
    supervisor = SupervisorAgent()
    print("Init Done.")

    # Dynamic whitelist managed by DynamicWhitelist multi-factor scoring.
    # Falls back to static list if dynamic scoring fails.
    fallback_symbols = [
        "SUI/USDT", "TIA/USDT", "LDO/USDT", "AAVE/USDT", "DYDX/USDT",
        "MKR/USDT", "RENDER/USDT", "WLD/USDT",
        "TON/USDT", "POL/USDT", "ONDO/USDT", "PENDLE/USDT", "XLM/USDT", "HBAR/USDT"
    ]

    def get_symbols():
        try:
            from src.quant.dynamic_whitelist import DynamicWhitelist
            dyn = DynamicWhitelist(exchange_id='hyperliquid', top_n=8, update_interval_hours=4)
            wl = dyn.get_whitelist()
            if wl:
                return wl
        except Exception as e:
            log.warning(f"Dynamic whitelist failed: {e}, using fallback")
        return fallback_symbols

    print("Init Settlement...")
    # Monitor ALL positions for profit booking (not just HL executor's symbols)
    # This ensures positions from perp_ls, pairs_arb also get TP/SL management
    settlement = SettlementAgent(owned_symbols=None)
    
    # Aggressive 60s interval
    interval = 60

    log.info(f"Starting AGGRESSIVE Hyperliquid Executor with Profit Booking...")
    
    async def trading_loop():
        while True:
            try:
                symbols = get_symbols()
                log.info(f"Trading {len(symbols)} symbols: {symbols}")
                for symbol in symbols:
                    await supervisor.run_cycle(symbol, exchange_id="hyperliquid")
                    await asyncio.sleep(1) # Fast sweep
                log.info(f"Scan complete. Active positions being managed by Settlement Agent. Waiting {interval}s...")
                await asyncio.sleep(interval)
            except Exception as e:
                log.error(f"Trading Loop Error: {e}")
                await asyncio.sleep(10)

    # Run Trading and Settlement in parallel
    try:
        await asyncio.gather(
            trading_loop(),
            settlement.run_forever()
        )
    finally:
        # Release FDs/sockets on shutdown only, not every sweep iteration
        try:
            await supervisor.execution_agent.multi_client.close()
        except Exception as close_err:
            log.error(f"Error closing multi-client on shutdown: {close_err}")

if __name__ == "__main__":
    asyncio.run(run_parallel())
