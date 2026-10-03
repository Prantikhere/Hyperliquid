import os
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"))

from src.execution.hl_raw import HlSdkClient
from src.utils.logger import log

def clean_all_positions():
    client = HlSdkClient()
    state = client.info.user_state(client.wallet_address)
    positions = [p for p in state.get('assetPositions', []) if float(p['position']['szi']) != 0]

    print(f"Found {len(positions)} open positions to clean up:")
    for p in positions:
        pos = p['position']
        coin = pos['coin']
        szi = float(pos['szi'])
        upnl = float(pos.get('unrealizedPnl', 0))
        print(f"  {coin}: size={szi}, uPnL=${upnl:.2f}")

    for p in positions:
        pos = p['position']
        coin = pos['coin']
        szi = float(pos['szi'])
        if szi == 0:
            continue

        is_buy = (szi < 0)  # If short, buy to close; if long, sell to close
        qty = abs(szi)
        print(f"\nClosing {coin}: {'BUY' if is_buy else 'SELL'} {qty} (reduce_only=True)...")
        res = client.place_market_order(coin, is_buy=is_buy, qty=qty, slippage=0.03, reduce_only=True)
        print(f"Result for {coin}: {res}")
        time.sleep(1)

    print("\nVerifying post-cleanup positions...")
    time.sleep(2)
    new_state = client.info.user_state(client.wallet_address)
    new_positions = [p for p in new_state.get('assetPositions', []) if float(p['position']['szi']) != 0]
    margin = new_state.get('marginSummary', {})
    print(f"Remaining open positions: {len(new_positions)}")
    print(f"Account Value: ${float(margin.get('accountValue', 0)):.2f}")
    print(f"Total Margin Used: ${float(margin.get('totalMarginUsed', 0)):.2f}")
    print(f"Free Margin: ${float(margin.get('accountValue', 0)) - float(margin.get('totalMarginUsed', 0)):.2f}")

if __name__ == "__main__":
    clean_all_positions()
