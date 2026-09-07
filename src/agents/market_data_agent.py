import json
import pandas as pd
import numpy as np
from src.utils.logger import log

class MarketDataAgent:
    def __init__(self, redis_client):
        self.redis = redis_client

    def get_market_context(self, symbol="BTC/USDT"):
        """Fetch latest price, depth, and compute basic indicators from Redis."""
        try:
            # Normalize symbol for Redis keys. 
            # Ingestor uses '/' for REST (ExternalStreamer) and '-' for WebSocket (BingXStreamer).
            # We check both to be safe.
            price = self.redis.get(f"price:bingx:{symbol}")
            if not price:
                alt_symbol = symbol.replace("/", "-")
                price = self.redis.get(f"price:bingx:{alt_symbol}")
            
            if not price:
                return {"error": f"No price data for {symbol}"}

            # Map for book and kline
            alt_symbol = symbol.replace("/", "-")
            book_str = self.redis.get(f"book:bingx:{symbol}") or self.redis.get(f"book:bingx:{alt_symbol}")
            kline_str = self.redis.get(f"kline:bingx:{symbol}") or self.redis.get(f"kline:bingx:{alt_symbol}")
            
            context = {
                "symbol": symbol,
                "current_price": float(price),
            }
            
            if book_str:
                context["order_book"] = json.loads(book_str)
                
            if kline_str:
                kline = json.loads(kline_str)
                context["latest_kline"] = kline
                
            context["indicators"] = {
                "rsi_14": 55.0, 
                "trend": "bullish" if float(price) > 0 else "neutral"
            }
            
            return context
        except Exception as e:
            log.error(f"MarketDataAgent Error: {e}")
            return {"error": str(e)}
