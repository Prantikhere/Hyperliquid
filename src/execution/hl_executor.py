import asyncio
import os
import sys
import redis

# Ensure project root is on sys.path so 'src.*' imports work even when
# PYTHONPATH is not exported (e.g. watchdog nohup restarts).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv
from src.utils.logger import log
from src.agents.supervisor import SupervisorAgent
from src.agents.settlement_agent import SettlementAgent
from src.autonomous.brain import AutonomousBrain

load_dotenv()

async def run_parallel():
    print("Init Supervisor...")
    supervisor = SupervisorAgent()
    print("Init Done.")

    # Initialize Autonomous Brain
    print("Init Autonomous Brain...")
    try:
        r = redis.Redis(host=os.getenv('REDIS_HOST', 'localhost'), port=6379, decode_responses=True)
        brain = AutonomousBrain(
            db=supervisor.execution_agent.db if hasattr(supervisor.execution_agent, 'db') else None,
            redis_client=r,
            learning_module=supervisor.learning_module if hasattr(supervisor, 'learning_module') else None
        )
        # Inject brain into supervisor for trade recommendations
        supervisor.brain = brain
        print("Autonomous Brain initialized.")
    except Exception as e:
        log.error(f"Failed to init Autonomous Brain: {e}")
        brain = None

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
    settlement = SettlementAgent(owned_symbols=None, brain=brain)
    
    # Aggressive 60s interval
    interval = 60

    log.info(f"Starting AGGRESSIVE Hyperliquid Executor with Profit Booking...")
    
    async def trading_loop():
        cycle_count = 0
        while True:
            try:
                symbols = get_symbols()
                log.info(f"Trading {len(symbols)} symbols: {symbols}")
                for symbol in symbols:
                    await supervisor.run_cycle(symbol, exchange_id="hyperliquid")
                    await asyncio.sleep(1) # Fast sweep
                
                # Brain cycle every 5 trading loops (approx every 5 min)
                cycle_count += 1
                if brain and cycle_count % 5 == 0:
                    brain.cycle()
                    if cycle_count % 30 == 0:  # Log report every ~30 min
                        brain.log_report()
                
                log.info(f"Scan complete. Active positions being managed by Settlement Agent. Waiting {interval}s...")
                await asyncio.sleep(interval)
            except Exception as e:
                log.error(f"Trading Loop Error: {e}")
                await asyncio.sleep(10)

    # Run Trading and Settlement in parallel.
    # return_exceptions=True prevents one task crashing from killing the other.
    try:
        results = await asyncio.gather(
            trading_loop(),
            settlement.run_forever(),
            return_exceptions=True
        )
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                log.error(f"Task {i} exited with error: {result}")
                import traceback
                traceback.print_exception(type(result), result, result.__traceback__)
    except Exception as e:
        log.error(f"Fatal parallel error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Release FDs/sockets on shutdown only, not every sweep iteration
        try:
            await supervisor.execution_agent.multi_client.close()
        except Exception as close_err:
            log.error(f"Error closing multi-client on shutdown: {close_err}")

if __name__ == "__main__":
    import signal
    import logging

    def _handle_signal(signum, frame):
        log.warning(f"Received signal {signum} ({signal.Signals(signum).name}). Shutting down gracefully.")
        # Don't exit — let asyncio handle cleanup via the finally block

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    try:
        asyncio.run(run_parallel())
    except KeyboardInterrupt:
        log.warning("KeyboardInterrupt received. Shutting down.")
    except Exception as e:
        log.critical(f"hl_executor.py CRASHED: {e}")
        import traceback
        traceback.print_exc()
    finally:
        log.info("hl_executor.py process exiting.")
