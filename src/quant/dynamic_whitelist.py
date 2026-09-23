"""
Dynamic Whitelist Generator
Multi-factor scoring system for automatic token selection.
Scores tokens on: Volume, Momentum, Volatility, Funding Rate, Spread.
Updates whitelist every N hours based on current market conditions.
"""
import time
import json
import os
import numpy as np
from src.utils.logger import log
from src.utils.db import DatabaseManager
import redis


class DynamicWhitelist:
    """
    Generates a dynamic whitelist of tradable tokens based on multi-factor scoring.
    
    Scoring Factors:
    - Volume (30%): 24h volume relative to median
    - Momentum (25%): 7d price change
    - Volatility (20%): Recent price swings (good for entries)
    - Funding Rate (15%): Perpetual futures sentiment
    - Spread (10%): Bid-ask spread (liquidity)
    
    Usage:
        wl = DynamicWhitelist()
        whitelist = wl.get_whitelist()  # Returns list of symbol strings
    """
    
    # Scoring weights (must sum to 1.0)
    WEIGHTS = {
        'volume': 0.30,
        'momentum': 0.25,
        'volatility': 0.20,
        'funding': 0.15,
        'spread': 0.10
    }
    
    # Default static whitelist (fallback if dynamic scoring fails)
    # Keep in sync with models_local/strategy_config.json symbol_whitelist
    STATIC_FALLBACK = [
        "ARB/USDT", "ETC/USDT", "PENDLE/USDT", "ONDO/USDT",
        "OP/USDT", "INJ/USDT", "APT/USDT"
    ]
    
    # Minimum requirements for inclusion
    MIN_VOLUME_USD = 1_000_000  # $1M daily volume minimum
    MIN_PRICE_USD = 0.02  # Include low-price tokens like HBAR
    MAX_SPREAD_PCT = 0.5  # Max 0.5% spread
    
    def __init__(self, exchange_id='hyperliquid', top_n=8, update_interval_hours=4):
        self.exchange_id = exchange_id
        self.top_n = top_n
        self.update_interval = update_interval_hours * 3600
        self.redis = redis.Redis(host=os.getenv('REDIS_HOST', 'localhost'), port=6379, decode_responses=True)
        self.db = DatabaseManager()
        self.last_update_key = f"dynamic_whitelist:last_update:{exchange_id}"
        self.whitelist_key = f"dynamic_whitelist:symbols:{exchange_id}"
        self.scores_key = f"dynamic_whitelist:scores:{exchange_id}"
        
    def get_whitelist(self, force_update=False):
        """
        Get current whitelist. Updates automatically based on interval.
        Returns list of symbol strings like ["BTC/USDT", "ETH/USDT", ...]
        """
        try:
            # Check if we need to update
            last_update = float(self.redis.get(self.last_update_key) or 0)
            now = time.time()
            
            if not force_update and (now - last_update) < self.update_interval:
                # Use cached whitelist
                cached = self.redis.get(self.whitelist_key)
                if cached:
                    whitelist = json.loads(cached)
                    log.debug(f"[DYN_WHITELIST] Using cached whitelist ({len(whitelist)} symbols)")
                    return whitelist
            
            # Perform update
            log.info(f"[DYN_WHITELIST] Updating whitelist (force={force_update})...")
            whitelist = self._compute_whitelist()
            
            if whitelist and len(whitelist) >= 3:
                # Cache the result
                self.redis.set(self.whitelist_key, json.dumps(whitelist), ex=self.update_interval + 300)
                self.redis.set(self.last_update_key, str(now), ex=self.update_interval + 300)
                log.info(f"[DYN_WHITELIST] Updated: {whitelist}")
                return whitelist
            else:
                log.warning(f"[DYN_WHITELIST] Scoring failed or too few tokens, using static fallback")
                return self.STATIC_FALLBACK.copy()
                
        except Exception as e:
            log.error(f"[DYN_WHITELIST] Error getting whitelist: {e}")
            return self.STATIC_FALLBACK.copy()
    
    def _compute_whitelist(self):
        """Compute new whitelist based on multi-factor scoring."""
        try:
            # Get all available symbols from exchange
            symbols = self._get_available_symbols()
            if not symbols:
                return None

            log.info(f"[DYN_WHITELIST] Full universe: {len(symbols)} symbols")

            # --- FAST PRE-FILTER: Narrow to top-40 by volume using allMids prices ---
            # This avoids expensive per-symbol candle fetches for illiquid coins
            volumes_24h = self._fetch_24h_volumes()
            if volumes_24h and len(volumes_24h) > self.top_n * 2:
                # Sort by volume score and take top 40 candidates (3x final top_n)
                pre_filter_n = max(self.top_n * 3, 40)
                vol_ranked = sorted(
                    [(s, volumes_24h.get(s, 0.0)) for s in symbols],
                    key=lambda x: x[1], reverse=True
                )[:pre_filter_n]
                symbols = [s for s, _ in vol_ranked]
                log.info(f"[DYN_WHITELIST] Pre-filtered to top {len(symbols)} by volume for detailed scoring")

            # Fetch market data for each symbol
            market_data = self._fetch_market_data(symbols)
            if not market_data:
                return None
            
            # Score each symbol
            scores = {}
            skipped = 0
            for symbol, data in market_data.items():
                score = self._score_symbol(data)
                if score is not None:
                    scores[symbol] = score
                else:
                    skipped += 1
            
            log.info(f"[DYN_WHITELIST] Scored {len(scores)} symbols, skipped {skipped} (of {len(market_data)} with data)")
            if not scores:
                log.warning(f"[DYN_WHITELIST] No symbols scored — market_data had {len(market_data)} entries")
                return None
            
            # Filter out churn-blocked symbols (4+ exits today)
            try:
                import time as _time
                day_key = int(_time.time() / 86400)
                churn_blocked = set()
                for symbol in list(scores.keys()):
                    exit_key = f"exit_count:{self.exchange_id}:{symbol}:{day_key}"
                    exits = int(self.redis.get(exit_key) or 0)
                    if exits >= 4:
                        churn_blocked.add(symbol)
                        del scores[symbol]
                        log.debug(f"[DYN_WHITELIST] {symbol} excluded: {exits} exits today (churn-blocked)")
                if churn_blocked:
                    log.info(f"[DYN_WHITELIST] Excluded {len(churn_blocked)} churn-blocked symbols: {churn_blocked}")
            except Exception as e:
                log.debug(f"[DYN_WHITELIST] Churn filter skipped: {e}")
            
            # Sort by score and take top N
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            top_symbols = [symbol for symbol, score in ranked[:self.top_n]]
            
            # Log scores for debugging
            log.info(f"[DYN_WHITELIST] Top {self.top_n} scores:")
            for i, (symbol, score) in enumerate(ranked[:self.top_n]):
                log.info(f"  {i+1}. {symbol}: {score:.3f}")
            
            # Store scores for monitoring
            self.redis.set(self.scores_key, json.dumps(dict(ranked[:20])), ex=self.update_interval)
            
            return top_symbols
            
        except Exception as e:
            log.error(f"[DYN_WHITELIST] Error computing whitelist: {e}")
            return None
    
    # Permanently excluded symbols — hardblock even if they appear in SDK universe
    EXCLUDED_SYMBOLS = {"NEAR/USDT", "FIL/USDT", "HBAR/USDT"}

    # Low-quality/meme coins not suitable for directional trading
    MEME_EXCLUSIONS = {
        "FARTCOIN/USDT", "JELLYJELLY/USDT", "JELLY/USDT", "USELESS/USDT",
        "CASHCAT/USDT", "CHILLGUY/USDT", "BANANA/USDT", "TRUMPCOIN/USDT",
        "MELANIA/USDT", "TST/USDT", "TRUMP/USDT", "WLFI/USDT", "YZY/USDT",
        "LAUNCHCOIN/USDT", "PUMP/USDT", "CHIP/USDT", "FRIEND/USDT",
        "HPOS/USDT", "RLB/USDT", "UNIBOT/USDT", "kBONK/USDT", "kDOGS/USDT",
        "kLUNC/USDT", "kNEIRO/USDT", "kPEPE/USDT", "kSHIB/USDT",
        "VINE/USDT", "DOOD/USDT", "PURR/USDT", "CC/USDT", "PONS/USDT",
    }

    def _get_available_symbols(self):
        """
        Get list of tradable USDT perpetual symbols from Hyperliquid.
        Primary source: HL SDK all_mids() — covers the full 200+ coin universe.
        Secondary: DB external_prices — confirms oracle prices are flowing.
        """
        # --- PRIMARY: HL SDK universe (real-time, covers full 200+ coin universe) ---
        sdk_symbols = []
        try:
            from src.execution.hl_raw import HlSdkClient
            client = HlSdkClient()
            mids = client.info.all_mids()
            coins = list(client.coin_to_asset.keys())
            # Only coins that have a live price (oracle is active)
            for coin in coins:
                mid = mids.get(coin)
                if mid:
                    try:
                        price = float(mid)
                        if price > 0:
                            symbol = f"{coin}/USDT"
                            sdk_symbols.append(symbol)
                    except (ValueError, TypeError):
                        continue
            log.info(f"[DYN_WHITELIST] SDK universe: {len(sdk_symbols)} active symbols")
        except Exception as e:
            log.warning(f"[DYN_WHITELIST] SDK universe query failed: {e}")

        # --- SECONDARY: DB-confirmed symbols (active in last 2 hours) ---
        db_symbols = set()
        try:
            query = """
                SELECT DISTINCT symbol 
                FROM external_prices 
                WHERE symbol LIKE '%/USDT' 
                AND time > NOW() - INTERVAL '2 hours'
                ORDER BY symbol
            """
            result = self.db.execute_query(query)
            if result:
                for row in result:
                    s = row[0]
                    if s.endswith('/USDT') and not any(x in s for x in ['_PERP', 'DOWN', 'UP']):
                        if ':' in s:
                            s = s.split(':', 1)[1]
                        db_symbols.add(s)
            log.debug(f"[DYN_WHITELIST] DB-confirmed symbols: {len(db_symbols)}")
        except Exception as e:
            log.warning(f"[DYN_WHITELIST] DB symbol query failed: {e}")

        # --- MERGE: prefer SDK universe; filter out excluded and meme coins ---
        all_excluded = self.EXCLUDED_SYMBOLS | self.MEME_EXCLUSIONS
        if sdk_symbols:
            candidates = [s for s in sdk_symbols if s not in all_excluded]
        else:
            # SDK unavailable — fall back to DB-confirmed symbols only
            candidates = [s for s in db_symbols if s not in all_excluded]

        log.info(f"[DYN_WHITELIST] Candidate universe after exclusions: {len(candidates)} symbols")
        return candidates
    
    def _fetch_market_data(self, symbols):
        """
        Fetch market data for scoring.
        Primary: HL SDK candle/price data (works for all 200+ universe symbols).
        Secondary: DB historical data (only for actively-streamed symbols).
        """
        market_data = {}

        # Get current oracle prices and volume scores from exchange
        volumes_24h = self._fetch_24h_volumes()

        # Get live prices from SDK (covers full universe)
        sdk_prices = {}
        try:
            from src.execution.hl_raw import HlSdkClient
            client = HlSdkClient()
            mids = client.info.all_mids()
            for coin, mid in mids.items():
                try:
                    sdk_prices[f"{coin}/USDT"] = float(mid)
                except (ValueError, TypeError):
                    pass
        except Exception as e:
            log.warning(f"[DYN_WHITELIST] SDK price fetch failed: {e}")

        for symbol in symbols:
            try:
                coin = symbol.replace('/USDT', '')
                current_price = sdk_prices.get(symbol, 0)
                if current_price <= 0:
                    continue  # No live price — skip

                # --- Try DB first for historical bars (streamed symbols only) ---
                prices = None
                for prefix in [self.exchange_id, 'bingx']:
                    prefixed = f"{prefix}:{symbol}"
                    query = """
                        SELECT price FROM external_prices
                        WHERE symbol = %s
                        ORDER BY time DESC LIMIT 200
                    """
                    result = self.db.execute_query(query, (prefixed,))
                    if result and len(result) >= 10:
                        prices = [float(row[0]) for row in result]
                        break

                # --- SDK candle fallback for new symbols ---
                if not prices or len(prices) < 10:
                    try:
                        # Use HL candlestick endpoint for 1h candles (last 7 days = 168 bars)
                        import requests, time as _t
                        resp = requests.post(
                            'https://api.hyperliquid-testnet.xyz/info',
                            json={'type': 'candleSnapshot', 'req': {
                                'coin': coin, 'interval': '1h',
                                'startTime': int((_t.time() - 7*86400)*1000)
                            }},
                            timeout=8
                        )
                        if resp.status_code == 200:
                            candles = resp.json()
                            if candles and isinstance(candles, list):
                                prices = [float(c['c']) for c in candles if 'c' in c]
                    except Exception as ce:
                        log.debug(f"[DYN_WHITELIST] {symbol} candle fetch error: {ce}")

                if not prices or len(prices) < 5:
                    # Minimal data: use only current price with neutral scores
                    prices = [current_price]

                data = {
                    'symbol': symbol,
                    'current_price': current_price,
                    'prices': prices,
                    'volume_score': volumes_24h.get(symbol, 0.3),  # Default 0.3 for unscored
                }

                # Momentum (up to 7d, use all available history)
                if len(prices) >= 2:
                    old_price = prices[min(len(prices)-1, 167)]  # 168h = 7d of hourly
                    data['momentum_7d'] = (prices[0] - old_price) / old_price if old_price else 0
                else:
                    data['momentum_7d'] = 0

                # Volatility (std dev of returns)
                if len(prices) >= 3:
                    returns = np.diff(prices) / (np.array(prices[:-1]) + 1e-9)
                    data['volatility'] = float(np.std(returns[-50:]))
                else:
                    data['volatility'] = 0.02  # Assume moderate vol for new symbols

                # Spread (estimate from short price history)
                data['spread_score'] = self._estimate_spread(symbol)

                market_data[symbol] = data

            except Exception as e:
                log.debug(f"[DYN_WHITELIST] {symbol} data fetch error: {e}")
                continue

        log.info(f"[DYN_WHITELIST] Fetched market data for {len(market_data)}/{len(symbols)} symbols")
        return market_data
    
    def _estimate_volume(self, symbol):
        """Estimate relative volume (0-1 scale)."""
        try:
            # Count recent trades as volume proxy
            prefixed = f"{self.exchange_id}:{symbol}"
            query = """
                SELECT COUNT(*) 
                FROM system_trades 
                WHERE symbol = %s 
                AND time > NOW() - INTERVAL '24 hours'
            """
            result = self.db.execute_query(query, (prefixed,))
            if result:
                count = result[0][0]
                # Normalize: 100+ trades = full score
                return min(count / 100, 1.0)
            return 0.1
        except:
            return 0.1
    
    def _fetch_24h_volumes(self):
        """
        Fetch actual 24h quote volume from HL exchange for all symbols.
        Uses sync requests (safe inside or outside asyncio event loops).
        Returns dict: {symbol: normalized_volume_score (0-1)}
        """
        import requests, math
        volumes = {}
        try:
            # Use metaAndAssetCtxs for real 24h notional volume (dayNtlVlm)
            resp = requests.post(
                'https://api.hyperliquid-testnet.xyz/info',
                json={'type': 'metaAndAssetCtxs'},
                timeout=12
            )
            if resp.status_code != 200:
                raise ValueError(f"HTTP {resp.status_code}")

            data = resp.json()
            # data = [meta_dict, [assetCtx_dict, ...]]
            if not isinstance(data, list) or len(data) < 2:
                raise ValueError("Unexpected metaAndAssetCtxs format")

            meta = data[0]
            asset_ctxs = data[1]
            coins = [coin_info['name'] for coin_info in meta.get('universe', [])]

            volumes_raw = {}
            for coin, ctx in zip(coins, asset_ctxs):
                try:
                    # dayNtlVlm = 24h notional volume in USD
                    vol = float(ctx.get('dayNtlVlm', 0) or 0)
                    symbol = f"{coin}/USDT"
                    volumes_raw[symbol] = vol
                except (ValueError, TypeError, KeyError):
                    continue

            # Log-normalize to 0-1 scale
            if volumes_raw:
                max_vol = max(volumes_raw.values())
                if max_vol > 0:
                    for symbol, vol in volumes_raw.items():
                        normalized = math.log10(vol + 1) / math.log10(max_vol + 1)
                        volumes[symbol] = normalized

            log.info(f"[DYN_WHITELIST] Fetched real 24h volumes for {len(volumes)} symbols from exchange")
            return volumes

        except Exception as e:
            log.warning(f"[DYN_WHITELIST] Failed to fetch 24h volumes: {e}")
            # Fallback: use allMids prices as rough volume proxy (higher price = more liquid)
            try:
                resp = requests.post(
                    'https://api.hyperliquid-testnet.xyz/info',
                    json={'type': 'allMids'}, timeout=8
                )
                if resp.status_code == 200:
                    mids = resp.json()
                    volumes_raw = {}
                    for coin, mid_px in mids.items():
                        try:
                            price = float(mid_px)
                            if price > 0:
                                volumes_raw[f"{coin}/USDT"] = price
                        except (ValueError, TypeError):
                            continue
                    if volumes_raw:
                        max_v = max(volumes_raw.values())
                        if max_v > 0:
                            for sym, v in volumes_raw.items():
                                volumes[sym] = math.log10(v + 1) / math.log10(max_v + 1)
            except Exception as e2:
                log.warning(f"[DYN_WHITELIST] Fallback volume fetch also failed: {e2}")
            return volumes
    
    def _estimate_spread(self, symbol):
        """Estimate spread quality (0-1, higher = tighter spread)."""
        try:
            # Use recent price volatility as spread proxy
            prefixed = f"{self.exchange_id}:{symbol}"
            query = """
                SELECT price 
                FROM external_prices 
                WHERE symbol = %s 
                ORDER BY time DESC 
                LIMIT 10
            """
            result = self.db.execute_query(query, (prefixed,))
            if result and len(result) >= 5:
                prices = [float(row[0]) for row in result]
                spread = (max(prices) - min(prices)) / np.mean(prices)
                # Lower spread = higher score
                return max(0, 1 - spread * 10)  # 10% spread = 0 score
            return 0.5
        except:
            return 0.5
    
    def _score_symbol(self, data):
        """Calculate composite score for a symbol."""
        try:
            scores = {}
            
            # Volume score (already 0-1)
            scores['volume'] = data.get('volume_score', 0.1)
            
            # Momentum score (normalize to 0-1)
            momentum = data.get('momentum_7d', 0)
            # Map -20% to +20% -> 0 to 1
            scores['momentum'] = np.clip((momentum + 0.2) / 0.4, 0, 1)
            
            # Volatility score (moderate is best)
            vol = data.get('volatility', 0)
            # Sweet spot: 2-5% daily volatility
            if vol < 0.01:
                scores['volatility'] = 0.2  # Too quiet
            elif vol < 0.02:
                scores['volatility'] = 0.6
            elif vol < 0.05:
                scores['volatility'] = 1.0  # Ideal
            elif vol < 0.08:
                scores['volatility'] = 0.7
            else:
                scores['volatility'] = 0.3  # Too volatile
            
            # Funding rate score (neutral is best for mean reversion)
            # For now, use momentum as proxy
            scores['funding'] = 0.5  # Default neutral
            
            # Spread score (already 0-1)
            scores['spread'] = data.get('spread_score', 0.5)
            
            # Weighted composite
            composite = sum(scores[k] * self.WEIGHTS[k] for k in self.WEIGHTS)
            
            # Apply minimum price filter
            if data['current_price'] < self.MIN_PRICE_USD:
                log.debug(f"[DYN_WHITELIST] {data['symbol']} skipped: price ${data['current_price']:.4f} < ${self.MIN_PRICE_USD}")
                return None
            
            return composite
            
        except Exception as e:
            return None
    
    def get_scores(self):
        """Get current scores for monitoring."""
        try:
            cached = self.redis.get(self.scores_key)
            if cached:
                return json.loads(cached)
            return {}
        except:
            return {}
    
    def force_update(self):
        """Force immediate whitelist update."""
        return self.get_whitelist(force_update=True)
    
    def add_banned(self, symbol):
        """Add a symbol to banned list (never trade)."""
        banned_key = f"dynamic_whitelist:banned:{self.exchange_id}"
        banned = json.loads(self.redis.get(banned_key) or '[]')
        if symbol not in banned:
            banned.append(symbol)
            self.redis.set(banned_key, json.dumps(banned), ex=None)
            log.info(f"[DYN_WHITELIST] Banned {symbol}")
    
    def remove_banned(self, symbol):
        """Remove symbol from banned list."""
        banned_key = f"dynamic_whitelist:banned:{self.exchange_id}"
        banned = json.loads(self.redis.get(banned_key) or '[]')
        if symbol in banned:
            banned.remove(symbol)
            self.redis.set(banned_key, json.dumps(banned), ex=None)
            log.info(f"[DYN_WHITELIST] Unbanned {symbol}")


# Singleton instance
_dynamic_whitelist = None

def get_dynamic_whitelist(exchange_id='hyperliquid'):
    """Get singleton DynamicWhitelist instance."""
    global _dynamic_whitelist
    if _dynamic_whitelist is None:
        _dynamic_whitelist = DynamicWhitelist(exchange_id=exchange_id)
    return _dynamic_whitelist
