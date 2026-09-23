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

    # Static whitelist fallback (matches strategy_config.json / STATIC_FALLBACK)
    fallback_symbols = [
        "ARB/USDT", "ETC/USDT", "PENDLE/USDT", "ONDO/USDT",
        "OP/USDT", "INJ/USDT", "APT/USDT"
    ]

    def get_symbols():
        # Use supervisor's whitelist method which respects use_dynamic_whitelist config
        try:
            return supervisor._get_whitelist('hyperliquid')
        except Exception as e:
            log.warning(f"Whitelist failed: {e}, using fallback")
        return fallback_symbols

    print("Init Settlement...")
    # Restrict profit-booking to supervisor's own universe to prevent stomping on
    # market-neutral spreads from perp_ls or pairs_arb (same wallet).
    current_symbols = get_symbols()
    settlement = SettlementAgent(owned_symbols=current_symbols, brain=brain)
    
    # Aggressive 60s interval
    interval = 60

    log.info(f"Starting AGGRESSIVE Hyperliquid Executor with Profit Booking...")

    async def universe_feed_loop():
        """
        Background task: polls ALL Hyperliquid oracle prices every 30s.
        Writes to Redis (price:hyperliquid:{symbol}) AND DB (external_prices).
        This populates the universe so DynamicWhitelist can score all 200+ symbols.
        """
        from src.execution.hl_raw import HlSdkClient
        from src.utils.db import DatabaseManager
        from datetime import datetime, timezone
        import redis as _redis
        feed_db = DatabaseManager()
        feed_redis = _redis.Redis(host=os.getenv('REDIS_HOST', 'localhost'), port=6379, decode_responses=True)
        FEED_INTERVAL = 30  # seconds
        DB_WRITE_EVERY = 4   # write to DB every 4th cycle (~2 min) to avoid overwhelming it
        cycle = 0

        # Excluded toxic symbols — never write to Redis for trading consideration
        EXCLUDED = {"NEAR/USDT", "FIL/USDT", "HBAR/USDT",
                    "FARTCOIN/USDT", "JELLYJELLY/USDT", "JELLY/USDT", "USELESS/USDT"}

        log.info("[UNIVERSE_FEED] Starting full-universe price feed (30s interval)...")
        while True:
            try:
                client = HlSdkClient()
                mids = client.info.all_mids()
                now = datetime.now(timezone.utc)
                cycle += 1
                do_db_write = (cycle % DB_WRITE_EVERY == 0)

                pipe = feed_redis.pipeline()
                db_batch = []

                for coin, mid_str in mids.items():
                    try:
                        price = float(mid_str)
                        if price <= 0:
                            continue
                        symbol = f"{coin}/USDT"
                        if symbol in EXCLUDED:
                            continue

                        # Always write to Redis for real-time consumption
                        pipe.set(f"price:hyperliquid:{symbol}", str(price), ex=120)

                        # Write to DB periodically for scoring history
                        if do_db_write:
                            db_batch.append((now, f"hyperliquid:{symbol}", price, 0))
                    except (ValueError, TypeError):
                        continue

                pipe.execute()

                if do_db_write and db_batch:
                    for row in db_batch:
                        feed_db.insert_external_price(row)
                    feed_db.conn.commit()
                    log.debug(f"[UNIVERSE_FEED] Wrote {len(db_batch)} prices to DB")

                log.debug(f"[UNIVERSE_FEED] Updated {len(mids)} oracle prices in Redis")

            except Exception as e:
                log.warning(f"[UNIVERSE_FEED] Error: {e}")

            await asyncio.sleep(FEED_INTERVAL)


    
    async def trading_loop():
        cycle_count = 0
        while True:
            try:
                symbols = get_symbols()
                settlement.owned_symbols = set(symbols)  # Keep settlement universe synchronized
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

    # Run Trading, Settlement and Universe Feed in parallel.
    # return_exceptions=True prevents one task crashing from killing the other.
    try:
        results = await asyncio.gather(
            trading_loop(),
            settlement.run_forever(),
            universe_feed_loop(),
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
