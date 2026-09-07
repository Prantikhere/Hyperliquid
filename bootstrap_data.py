import asyncio
import ccxt.async_support as ccxt
import os
from src.utils.db import DatabaseManager
from src.utils.logger import log

async def bootstrap():
    db = DatabaseManager()
    log.info("Bootstrapping 24h historical data for 50 tokens...")
    
    # Representative 50 symbols
    symbols = [
        "BTC/USDT", "ETH/USDT", "BNB/USDT", "SOL/USDT", "XRP/USDT",
        "DOGE/USDT", "ADA/USDT", "TRX/USDT", "AVAX/USDT", "DOT/USDT",
        "LINK/USDT", "SHIB/USDT", "BCH/USDT", "NEAR/USDT", "LTC/USDT",
        "UNI/USDT", "PEPE/USDT", "STX/USDT", "KAS/USDT", "ICP/USDT",
        "APT/USDT", "RENDER/USDT", "HBAR/USDT", "ARB/USDT", "FIL/USDT",
        "ATOM/USDT", "VET/USDT", "MKR/USDT", "TIA/USDT", "FTM/USDT",
        "OP/USDT", "RUNE/USDT", "GRT/USDT", "LDO/USDT", "SUI/USDT",
        "INJ/USDT", "BEAM/USDT", "AAVE/USDT", "FLOKI/USDT", "GALA/USDT",
        "SEI/USDT", "JUP/USDT", "FET/USDT", "WIF/USDT", "STRK/USDT",
        "PYTH/USDT", "BGB/USDT", "PENDLE/USDT", "ENA/USDT", "ARKM/USDT"
    ]
    
    exchange = ccxt.bingx({'options': {'defaultType': 'swap'}})
    
    try:
        for symbol in symbols:
            try:
                log.info(f"Fetching 1h candles for {symbol}...")
                # Fetch last 100 hours of data for a solid regime foundation
                ohlcv = await exchange.fetch_ohlcv(symbol, timeframe='1h', limit=100)
                for candle in ohlcv:
                    # candle format: [timestamp, open, high, low, close, volume]
                    # We inject into external_prices to seed the RegimeEngine
                    db.insert_external_price((
                        pd.to_datetime(candle[0], unit='ms'), 
                        f"bingx:{symbol}", 
                        candle[4], 
                        candle[5]
                    ))
            except Exception as e:
                log.warning(f"Could not bootstrap {symbol}: {e}")
        log.info("Bootstrap complete. Regime Engine will now have data.")
    finally:
        await exchange.close()

if __name__ == "__main__":
    import pandas as pd
    asyncio.run(bootstrap())
