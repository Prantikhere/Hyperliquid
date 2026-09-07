import hmac
import hashlib
import time
import requests
import os
import json
from src.utils.logger import log

class BingXClient:
    def __init__(self):
        self.api_key = os.getenv("BINGX_API_KEY")
        self.secret_key = os.getenv("BINGX_SECRET_KEY")
        self.base_url = "https://open-api.bingx.com"
        
        if not self.api_key or not self.secret_key:
            log.warning("BINGX_API_KEY or BINGX_SECRET_KEY not found in environment variables.")

    def _generate_signature(self, params_str):
        return hmac.new(
            self.secret_key.encode("utf-8"),
            params_str.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

    def _request(self, method, path, params=None):
        if params is None:
            params = {}
        
        params["timestamp"] = int(time.time() * 1000)
        
        # Sort and join params for signature
        query_string = "&".join([f"{k}={v}" for k, v in sorted(params.items())])
        signature = self._generate_signature(query_string)
        url = f"{self.base_url}{path}?{query_string}&signature={signature}"
        
        headers = {
            "X-BX-APIKEY": self.api_key,
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.request(method, url, headers=headers)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            log.error(f"BingX API Error ({path}): {e}")
            if hasattr(e, 'response') and e.response:
                log.error(f"Response: {e.response.text}")
            return {"code": -1, "msg": str(e)}

    # --- Account & Position Endpoints (USDT-M) ---
    def get_balance(self):
        """Get account balance for Perpetual Futures (USDT-M)."""
        return self._request("GET", "/openApi/swap/v2/user/balance")

    def get_positions(self, symbol=None):
        """Get current positions."""
        params = {}
        if symbol:
            params["symbol"] = symbol
        return self._request("GET", "/openApi/swap/v2/user/positions", params)

    def set_leverage(self, symbol, leverage, side="BOTH"):
        """Set leverage for a symbol (Perpetual). side: LONG, SHORT, or BOTH."""
        params = {
            "symbol": symbol,
            "leverage": leverage,
            "side": side
        }
        return self._request("POST", "/openApi/swap/v2/trade/leverage", params)

    def set_margin_mode(self, symbol, margin_mode):
        """Set margin mode: ISOLATED or CROSS."""
        params = {
            "symbol": symbol,
            "marginMode": margin_mode
        }
        return self._request("POST", "/openApi/swap/v2/trade/marginType", params)

    # --- Market Data Endpoints ---
    def get_ticker(self, symbol):
        """Get 24hr ticker for a symbol."""
        return self._request("GET", "/openApi/swap/v2/quote/ticker", {"symbol": symbol})

    def get_klines(self, symbol, interval="1m", limit=100):
        """Get candlestick data."""
        params = {
            "symbol": symbol,
            "interval": interval,
            "limit": limit
        }
        return self._request("GET", "/openApi/swap/v2/quote/klines", params)

    def get_funding_rate(self, symbol):
        """Get current and predicted funding rate."""
        return self._request("GET", "/openApi/swap/v2/quote/premiumIndex", {"symbol": symbol})

    # --- Trade Endpoints ---
    def place_order(self, symbol, side, type, quantity, price=None, stop_loss=None, take_profit=None):
        """Place an order (Market or Limit)."""
        params = {
            "symbol": symbol,
            "side": side,  # BUY or SELL
            "type": type,  # LIMIT or MARKET
            "quantity": quantity,
        }
        if price:
            params["price"] = price
        
        # BingX Swap V2 uses specific parameters for TP/SL trigger in orders if supported
        # For simplicity, we implement them as separate trigger params or manual management in higher levels
        return self._request("POST", "/openApi/swap/v2/trade/order", params)

    def cancel_order(self, symbol, order_id):
        """Cancel an existing order."""
        params = {
            "symbol": symbol,
            "orderId": order_id
        }
        return self._request("POST", "/openApi/swap/v2/trade/order/cancel", params)

if __name__ == "__main__":
    client = BingXClient()
    print(json.dumps(client.get_ticker("BTC-USDT"), indent=2))
