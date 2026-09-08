#!/usr/bin/env python3
"""Close NEAR LONG position to free margin."""
import asyncio
import sys
import os
from pathlib import Path

# Ensure project root is in path
PROJECT = Path('/home/prantik/Downloads/Personal/TradingBingx')
sys.path.insert(0, str(PROJECT))
os.chdir(PROJECT)

# Load environment
from dotenv import load_dotenv
load_dotenv(PROJECT / '.env')

from src.execution.multi_client import MultiExchangeClient

async def close_near():
    client = MultiExchangeClient()
    
    try:
        # Get NEAR price
        near_ticker = await client.hl.fetch_ticker('NEAR/USDC:USDC')
        near_price = near_ticker['last']
        print(f'NEAR current price: ${near_price}')
        
        # Close NEAR LONG (sell to close)
        print('Closing NEAR LONG (13.6 contracts)...')
        near_result = await client.place_order(
            'hyperliquid', 'NEAR/USDT', 'sell', 'MARKET', 13.6,
            price=near_price, reduce_only=True
        )
        print(f'NEAR result: {near_result}')
        
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
    finally:
        await client.hl.close()

if __name__ == '__main__':
    asyncio.run(close_near())
