import asyncio
import os
from dotenv import load_dotenv
from src.execution.multi_client import MultiExchangeClient
from openai import AsyncOpenAI

load_dotenv()

async def test_all():
    print("Testing MIMO API Key...")
    client = AsyncOpenAI(api_key=os.getenv("MIMO_API_KEY"), base_url=os.getenv("MIMO_BASE_URL"))
    try:
        response = await client.chat.completions.create(
            model=os.getenv("MIMO_MODEL"),
            messages=[{"role": "user", "content": "Ping"}],
            timeout=10.0
        )
        print("MIMO LLM API: OK")
    except Exception as e:
        print(f"MIMO LLM API ERROR: {e}")

    print("\nTesting Exchange Keys...")
    exchanges = MultiExchangeClient()
    try:
        hl_bal = await exchanges.get_balance("hyperliquid")
        if "error" not in str(hl_bal).lower():
            print("Hyperliquid API: OK")
        else:
            print(f"Hyperliquid API WARNING: {hl_bal}")
            
        bx_bal = await exchanges.get_balance("bingx")
        if "error" not in str(bx_bal).lower():
            print("BingX API: OK")
        else:
            print(f"BingX API WARNING: {bx_bal}")
    except Exception as e:
        print(f"Exchange API ERROR: {e}")
    finally:
        await exchanges.close()

if __name__ == "__main__":
    asyncio.run(test_all())
