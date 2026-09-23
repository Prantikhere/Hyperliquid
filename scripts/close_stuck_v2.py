"""
Emergency close script v2 - uses limit orders within oracle range.
"""
import os
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

from src.execution.hl_raw import HlSdkClient

def close_position_limit(client: HlSdkClient, coin: str):
    """Close position using limit order at oracle price."""
    print(f"\n--- Closing {coin} ---")
    
    pos = client.get_position(coin)
    if not pos:
        print(f"  No position found for {coin}")
        return
    
    szi = float(pos.get("szi", 0))
    entry_px = float(pos.get("entryPx", 0))
    
    print(f"  Position: {szi} {coin} (entry={entry_px})")
    
    if szi == 0:
        print(f"  Position already flat")
        return
    
    # Get current oracle price
    oracle_px = client._get_oracle_px(coin)
    print(f"  Oracle price: {oracle_px}")
    
    # Determine close side
    is_buy = szi < 0  # Close short = buy back
    close_qty = abs(szi)
    
    # Try limit order at oracle price (within HL tolerance)
    # HL allows orders within ~1-2% of oracle
    limit_px = oracle_px  # Use exact oracle price
    rounded_px = client._round_price(coin, limit_px)
    
    print(f"  Limit order: {'BUY' if is_buy else 'SELL'} {close_qty} @ {rounded_px} (oracle={oracle_px})")
    
    result = client.place_limit_order(
        coin=coin,
        is_buy=is_buy,
        qty=close_qty,
        price=rounded_px,
        reduce_only=True,
        tif="Gtc"  # Good till cancelled
    )
    
    print(f"  Result: {result}")
    
    if result.get("status") in ("filled", "resting"):
        print(f"  Order placed/filled for {coin}!")
        # Wait a moment and check if filled
        time.sleep(2)
        pos_after = client.get_position(coin)
        if pos_after:
            szi_after = float(pos_after.get("szi", 0))
            print(f"  Position after: {szi_after}")
        else:
            print(f"  Position closed!")
    elif "error" in result:
        print(f"  FAILED: {result['error']}")

def main():
    print("=" * 60)
    print("Emergency Close Stuck Positions (v2 - Limit Orders)")
    print("=" * 60)
    
    client = HlSdkClient()
    
    # Check current positions
    positions = client.get_positions()
    print(f"\nOpen positions: {len(positions)}")
    for pos in positions:
        p = pos.get("position", {})
        coin = p.get("coin", "?")
        szi = float(p.get("szi", 0))
        entry = p.get("entryPx", "?")
        print(f"  {coin}: {szi} (entry={entry})")
    
    # Close NEAR (the stuck one)
    close_position_limit(client, "NEAR")
    
    # Verify
    print("\n--- After Close ---")
    positions = client.get_positions()
    for pos in positions:
        p = pos.get("position", {})
        coin = p.get("coin", "?")
        szi = float(p.get("szi", 0))
        print(f"  {coin}: {szi}")
    
    balance = client.get_balance()
    print(f"\nFinal Balance: ${balance['account_value']:.2f}")

if __name__ == "__main__":
    main()
