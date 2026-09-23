"""
Emergency close script for stuck positions on Hyperliquid testnet.
Uses the official HL SDK directly.
"""
import os
import sys

# Ensure project root is on sys.path
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

from src.execution.hl_raw import HlSdkClient

def close_position(client: HlSdkClient, coin: str):
    """Close an open position using limit order with wider slippage."""
    print(f"\n--- Closing {coin} ---")
    
    pos = client.get_position(coin)
    if not pos:
        print(f"  No position found for {coin}")
        return
    
    szi = float(pos.get("szi", 0))
    entry_px = float(pos.get("entryPx", 0))
    current_value = abs(szi) * entry_px
    
    print(f"  Position: {szi} {coin} (entry={entry_px})")
    print(f"  Value: ~${current_value:.2f}")
    
    if szi == 0:
        print(f"  Position already flat")
        return
    
    # Determine close side: if long (szi > 0), sell; if short (szi < 0), buy
    is_buy = szi < 0  # Close short = buy back
    close_qty = abs(szi)
    
    # Get current oracle price
    oracle_px = client._get_oracle_px(coin)
    print(f"  Oracle price: {oracle_px}")
    
    # Use wide slippage for urgent close (5%)
    slippage = 0.05
    if is_buy:
        ioc_px = client._round_price(coin, oracle_px * (1 + slippage))
    else:
        ioc_px = client._round_price(coin, oracle_px * (1 - slippage))
    
    print(f"  Closing: {'BUY' if is_buy else 'SELL'} {close_qty} @ {ioc_px} (oracle={oracle_px})")
    
    result = client.place_market_order(
        coin=coin,
        is_buy=is_buy,
        qty=close_qty,
        slippage=slippage,
        reduce_only=True
    )
    
    print(f"  Result: {result}")
    
    if result.get("status") == "filled":
        print(f"  SUCCESS: {coin} position closed!")
    elif "error" in result:
        print(f"  FAILED: {result['error']}")
        # Try with even wider slippage (10%)
        print(f"  Retrying with 10% slippage...")
        slippage = 0.10
        if is_buy:
            ioc_px = client._round_price(coin, oracle_px * (1 + slippage))
        else:
            ioc_px = client._round_price(coin, oracle_px * (1 - slippage))
        
        result = client.place_market_order(
            coin=coin,
            is_buy=is_buy,
            qty=close_qty,
            slippage=slippage,
            reduce_only=True
        )
        print(f"  Retry result: {result}")

def main():
    print("=" * 60)
    print("Emergency Close Stuck Positions")
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
    
    # Close stuck positions
    close_position(client, "NEAR")
    close_position(client, "HBAR")
    
    # Verify
    print("\n--- After Close ---")
    positions = client.get_positions()
    for pos in positions:
        p = pos.get("position", {})
        coin = p.get("coin", "?")
        szi = float(p.get("szi", 0))
        print(f"  {coin}: {szi}")
    
    # Show final balance
    balance = client.get_balance()
    print(f"\nFinal Balance: ${balance['account_value']:.2f}")
    print(f"Margin Used: ${balance['margin_used']:.2f}")
    print(f"Available: ${balance['available']:.2f}")

if __name__ == "__main__":
    main()
