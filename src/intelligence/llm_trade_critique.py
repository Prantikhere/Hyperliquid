"""
SHADOW-ONLY LLM trade critique. Batch job (run via cron), NOT imported by any live process.
Reads recent closed trades, asks an LLM for qualitative critique, writes to a human-reviewed
log file. Never writes back to the DB, never touches supervisor.py or any decision path --
the "QUANT-ONLY DECISION" policy in supervisor.py is untouched by this file.
"""
import asyncio
import json
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

from src.utils.db import DatabaseManager
from src.utils.llm import FallbackLLMClient
from src.utils.logger import log

CRITIQUE_LOG = "logs/llm_critique.log"
LOOKBACK_DAYS = 7
MAX_TRADES = 40


def fetch_recent_trades(db, days=LOOKBACK_DAYS, limit=MAX_TRADES):
    query = """
    SELECT market_id, side, price, time, metadata FROM system_trades
    WHERE status = 'LIVE_OK'
    AND (metadata->>'outcome') IS NOT NULL
    AND time >= %s
    ORDER BY time DESC
    LIMIT %s
    """
    since = datetime.now(timezone.utc) - timedelta(days=days)
    rows = db.execute_query(query, (since, limit))
    trades = []
    for market_id, side, price, time, meta in rows or []:
        trades.append({
            "symbol": market_id,
            "side": side,
            "price": float(price),
            "time": str(time),
            "regime": meta.get("regime"),
            "quant_action": meta.get("quant_action"),
            "meta_confidence": meta.get("meta_confidence"),
            "outcome_roi": meta.get("outcome"),
        })
    return trades


def build_prompt(trades):
    wins = sum(1 for t in trades if (t["outcome_roi"] or 0) > 0)
    losses = len(trades) - wins
    system = (
        "You are a trading-desk risk reviewer. You are given a batch of closed trades from a "
        "quant-only automated system (no LLM in the decision path). Critique patterns, don't "
        "restate the data. Be specific and skeptical -- call out regime/confidence mismatches, "
        "clustering of losses, and any structural risk you notice. This is advisory only and "
        "will NOT change how the bot trades; it is read by a human. Keep it under 300 words."
    )
    user = (
        f"Batch: {len(trades)} trades, {wins} wins / {losses} losses over last {LOOKBACK_DAYS} days.\n\n"
        f"{json.dumps(trades, indent=2, default=str)}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


async def run():
    db = DatabaseManager()
    trades = fetch_recent_trades(db)
    if len(trades) < 5:
        log.info(f"[LLM-CRITIQUE] Only {len(trades)} trades in last {LOOKBACK_DAYS}d, skipping (need >=5).")
        return

    client = FallbackLLMClient(allow=True)
    messages = build_prompt(trades)
    critique = await client.chat_completion(messages, temperature=0.3, max_tokens=800)

    header = f"\n{'='*80}\n[LLM-CRITIQUE] {datetime.now(timezone.utc).isoformat()}Z -- {len(trades)} trades reviewed\n{'='*80}\n"
    with open(CRITIQUE_LOG, "a") as f:
        f.write(header)
        f.write(critique if critique else "(all LLM providers failed -- no critique generated)")
        f.write("\n")

    if critique:
        log.info(f"[LLM-CRITIQUE] Wrote critique for {len(trades)} trades to {CRITIQUE_LOG}")
    else:
        log.warning("[LLM-CRITIQUE] All LLM providers failed, no critique written this cycle.")


if __name__ == "__main__":
    asyncio.run(run())
