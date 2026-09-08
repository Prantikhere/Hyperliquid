#!/usr/bin/env python3
"""Close FIL SHORT and NEAR LONG positions to free margin."""
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
from src.utils.logger import log

async def close_positions():
    client = MultiExchangeClient()
    
    try:
        # Get FIL price
        fil_ticker = await client.hl.fetch_ticker('FIL/USDC:USDC')
        fil_price = fil_ticker['last']
        print(f'FIL current price: ${fil_price}')
        
        # Close FIL SHORT (buy to close)
        print('Closing FIL SHORT (226.2 contracts)...')
        fil_result = await client.place_order(
            'hyperliquid', 'FIL/USDT', 'buy', 'MARKET', 226.2,
            price=fil_price, reduce_only=True
        )
        print(f'FIL result: {fil_result}')
        
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
        
        # Wait for settlement
        await asyncio.sleep(2)
        
        # Check balance
        balance = await client.hl.fetch_balance()
        info = balance.get('info', {})
        margin = info.get('marginSummary', {})
        print(f'\n=== MARGIN STATUS ===')
        print(f'Account Value: ${float(margin.get("accountValue", 0)):.2f}')
        print(f'Total Margin Used: ${float(margin.get("totalMarginUsed", 0)):.2f}')
        print(f'Free Margin: ${float(balance.get("free", {}).get("USDC", 0)):.2f}')
        
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
    finally:
        await client.hl.close()

if __name__ == '__main__':
    asyncio.run(close_positions())
