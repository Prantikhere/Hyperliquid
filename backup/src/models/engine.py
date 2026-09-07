import asyncio
import os
import redis
import json
import pandas as pd
from src.utils.db import DatabaseManager
from src.utils.logger import log
from src.models.feature_engine import FeatureEngine
from src.models.statistical import StatisticalModels
from src.models.ml_model import MLModel
from src.models.llm_sentiment import LLMSentiment

class MLEngine:
    def __init__(self):
        self.db = DatabaseManager()
        self.redis = redis.Redis(
            host=os.getenv('REDIS_HOST', 'redis'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            decode_responses=True
        )
        self.features = FeatureEngine(self.db)
        self.stat_models = StatisticalModels()
        self.ml_model = MLModel()
        self.llm = LLMSentiment()

    async def run(self):
        log.info("Starting AI/ML Ensemble Engine (Dual-Sided + Binance Synced)...")
        while True:
            try:
                # 1. Get all active markets from DB
                markets = self.db.execute_query("SELECT market_id, tokens, condition_id, question FROM markets WHERE active = TRUE")
                if not markets:
                    await asyncio.sleep(10)
                    continue
                
                # 2. Get latest Binance prices for comparison
                binance_btc = self.get_latest_price("binance:BTC/USDT")
                binance_eth = self.get_latest_price("binance:ETH/USDT")
                
                for market_id, tokens, condition_id, question in markets:
                    # 3. Determine if this market is BTC or ETH related
                    ref_price = binance_btc if "BTC" in question.upper() or "BITCOIN" in question.upper() else binance_eth
                    
                    # 4. Predict for outcome 0 (Yes)
                    await self.process_market_outcome(market_id, tokens, condition_id, ref_price)
                
                await asyncio.sleep(5)
            except Exception as e:
                log.error(f"Error in ML engine loop: {e}")
                await asyncio.sleep(10)

    def get_latest_price(self, symbol):
        try:
            val = self.redis.get(f"price:{symbol}")
            return float(val) if val else None
        except:
            return None

    async def process_market_outcome(self, market_id, tokens, condition_id, ref_price):
        # 1. Get features for token 0 (Yes)
        market_feats = self.features.get_market_features(market_id, outcome_index=0)
        if not market_feats:
            return

        try:
            clean_feats = {k: float(v) for k, v in market_feats.items()}
            
            # 2. ML Prediction
            ml_pred = self.ml_model.predict(clean_feats)
            
            # 3. Arbitrage / Correlation Signal
            # If we have a reference price (Binance), we can check for discrepancies
            # For a binary market, if BTC is pumping on Binance but not yet on Polymarket,
            # we increase the probability of YES.
            arb_adj = 0.0
            if ref_price:
                # Get previous ref price from DB to see the trend
                prev_ref = self.db.execute_query(f"SELECT price FROM external_prices WHERE symbol = 'binance:BTC/USDT' AND time < NOW() - INTERVAL '1 minute' ORDER BY time DESC LIMIT 1")
                if prev_ref:
                    change = (ref_price / float(prev_ref[0][0])) - 1.0
                    if change > 0.001: # 0.1% pump in 1 min
                        arb_adj = 0.05 # Add 5% to YES probability
                    elif change < -0.001:
                        arb_adj = -0.05
            
            # 4. Stat Signal
            stat_sig = self.stat_models.get_signal(clean_feats)
            stat_prob = 0.5
            if stat_sig:
                stat_prob = 0.5 + (0.3 * (1 if stat_sig['side'] == 'buy' else -1) * stat_sig['confidence'])
            
            # 5. Ensemble (Probability of YES)
            # Incorporating the ARB adjustment
            prob_yes = (0.3 * stat_prob) + (0.5 * ml_pred) + (0.2 * 0.5) + arb_adj
            prob_yes = max(0, min(1, prob_yes))
            
            log.info(f"Market {market_id} | Prob YES: {prob_yes:.4f} (ML: {ml_pred:.2f}, ArbAdj: {arb_adj:.2f}) | Ref: {ref_price}")
            
            # 6. Publish signals
            yes_signal = {"market_id": market_id, "condition_id": condition_id, "token_id": tokens[0], "outcome_index": 0, "probability": prob_yes}
            self.redis.set(f"market_signal:{market_id}:0", json.dumps(yes_signal))
            
            no_signal = {"market_id": market_id, "condition_id": condition_id, "token_id": tokens[1], "outcome_index": 1, "probability": 1.0 - prob_yes}
            self.redis.set(f"market_signal:{market_id}:1", json.dumps(no_signal))
            
        except Exception as e:
            log.error(f"Error processing outcome for {market_id}: {e}")

if __name__ == "__main__":
    engine = MLEngine()
    asyncio.run(engine.run())
