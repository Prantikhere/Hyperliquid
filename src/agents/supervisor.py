import asyncio
import os
import fcntl
import sys
import redis
import json
import pandas as pd
from src.agents.market_data_agent import MarketDataAgent
from src.agents.risk_agent import RiskAgent
from src.agents.execution_agent import ExecutionAgent
from src.utils.vector_store import VectorMemory
from src.utils.db import DatabaseManager
from src.quant.regime_engine import RegimeEngine
from src.quant.multi_strategy import StrategyEnsemble, RiskSurface
from src.intelligence.ensemble_model import EnsembleMetaLearner
from src.quant.anomaly_detector import AnomalyDetector
from src.utils.logger import log
from dotenv import load_dotenv

load_dotenv()

class SupervisorAgent:
    def __init__(self):
        self._acquire_singleton_lock()
        self.redis = redis.Redis(host=os.getenv('REDIS_HOST', 'localhost'), port=6379, decode_responses=True)
        self.db = DatabaseManager()
        self.market_agent = MarketDataAgent(self.redis)
        self.risk_agent = RiskAgent(db=self.db)
        self.execution_agent = ExecutionAgent(self.db)
        self.regime_engine = RegimeEngine()
        self.strategy_ensemble = StrategyEnsemble()
        self.risk_surface = RiskSurface()
        self.meta_learner = EnsembleMetaLearner()
        self.vector_memory = VectorMemory()  # shadow-only: recall/store, never touches decisions
        self.anomaly_detector = AnomalyDetector()  # shadow-only: logs outlier flag, never gates
        self.cfg = self._load_strategy_config()

    def _acquire_singleton_lock(self):
        """PID-file lock prevents duplicate supervisor instances from watchdog race."""
        lock_path = os.path.join(os.path.dirname(__file__), "..", "..", "state", "supervisor.pid")
        os.makedirs(os.path.dirname(lock_path), exist_ok=True)
        try:
            self._lock_fd = open(lock_path, "w")
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._lock_fd.write(str(os.getpid()))
            self._lock_fd.flush()
        except BlockingIOError:
            log.error("Another supervisor instance is already running. Exiting.")
            sys.exit(1)

    def _load_strategy_config(self):
        """Load backtested/tuned thresholds. Falls back to safe defaults if absent."""
        defaults = {"buy_threshold": 0.58, "sell_threshold": 0.42, "min_confidence": 0.45}
        try:
            with open("models_local/strategy_config.json") as f:
                cfg = json.load(f)
            log.info(f"Loaded backtested strategy config: {cfg}")
            return {**defaults, **cfg}
        except Exception:
            log.warning("No strategy_config.json; using default thresholds.")
            return defaults

    async def run_cycle(self, symbol, exchange_id="bingx"):
        log.info(f"--- [{exchange_id.upper()}] Full-Potential Cycle: {symbol} ---")
        
        # Dynamic Bankroll Syncing
        try:
            balance = await self.execution_agent.multi_client.get_balance(exchange_id)
            if balance:
                if exchange_id == 'hyperliquid':
                    equity = float(balance.get('info', {}).get('marginSummary', {}).get('accountValue', 0))
                else:
                    equity = float(balance.get('total', {}).get('USDT', 0) or balance.get('total', {}).get('USDC', 0) or 0)
                
                if equity > 0:
                    self.risk_agent.risk_manager.update_bankroll(equity)
                    log.info(f"[{exchange_id}] Dynamic bankroll updated to: ${equity:.2f}")
        except Exception as e:
            log.warning(f"Failed to update dynamic bankroll: {e}")

        # 1. Market Data
        price = self.redis.get(f"price:{exchange_id}:{symbol}") or self.redis.get(f"price:{symbol}")
        if not price: return {"action": "HOLD", "confidence": 0, "reason": "No price data"}

        # 2. Parallel Quant Execution
        # Query returns newest-first (ORDER BY time DESC). Reverse to chronological order
        # (oldest -> newest) so every downstream indicator (z-score, RSI, MACD proxy,
        # Kaufman efficiency ratio) is computed on a correctly-ordered time series.
        prices_raw = self.db.execute_query("SELECT price FROM external_prices WHERE symbol LIKE %s ORDER BY time DESC LIMIT 100", (f"%{symbol}%",))
        price_list = [p[0] for p in prices_raw][::-1] if prices_raw else [float(price)]
        
        regime = self.regime_engine.detect_regime(price_list)
        book = self.redis.get(f"book:{exchange_id}:{symbol}")
        quant_signals = self.strategy_ensemble.get_signals(price_list, json.loads(book) if book else None)
        var_risk = self.risk_surface.simulate_drawdown(100, 0.5)

        # SHADOW: structural-break anomaly score, independent of the rule-based DD-kill.
        # Never gates or sizes anything -- logged only, for future evaluation.
        try:
            import numpy as _anp
            _s = pd.Series(price_list, dtype=float)
            _ret = _s.pct_change().dropna()
            _tot = abs(_s.iloc[-1] - _s.iloc[0])
            _sum_abs = _s.diff().abs().sum()
            er = float(_tot / _sum_abs) if _sum_abs else 0.0
            _recent_n = max(5, len(_s) // 4)
            _full_vol = _ret.std()
            _recent_vol = _ret.tail(_recent_n).std()
            vol_ratio = float(_recent_vol / _full_vol) if _full_vol and _full_vol > 0 else 1.0
            is_anom, anom_score = self.anomaly_detector.score({
                **quant_signals, "er": er, "vol_ratio": vol_ratio
            })
            if is_anom:
                log.warning(f"[SHADOW-ANOMALY] {symbol}: structural-break outlier flagged (score={anom_score:.3f}, er={er:.2f}, vol_ratio={vol_ratio:.2f})")
        except Exception as e:
            log.debug(f"[SHADOW-ANOMALY] skipped: {e}")
        
        # Multi-timeframe trend calculation
        trend_1h, trend_4h = "Neutral", "Neutral"
        try:
            p_1h = self.db.execute_query("SELECT price FROM external_prices WHERE symbol LIKE %s AND time < NOW() - INTERVAL '1 hour' ORDER BY time DESC LIMIT 1", (f"%{symbol}%",))
            if p_1h:
                change = (float(price) - p_1h[0][0]) / p_1h[0][0]
                trend_1h = "Bullish" if change > 0.005 else "Bearish" if change < -0.005 else "Neutral"
                
            p_4h = self.db.execute_query("SELECT price FROM external_prices WHERE symbol LIKE %s AND time < NOW() - INTERVAL '4 hours' ORDER BY time DESC LIMIT 1", (f"%{symbol}%",))
            if p_4h:
                change = (float(price) - p_4h[0][0]) / p_4h[0][0]
                trend_4h = "Bullish" if change > 0.015 else "Bearish" if change < -0.015 else "Neutral"
        except Exception as e:
            log.warning(f"Trend Calc Error: {e}")
        
        # 3. Position Management
        current_pos = self.db.get_positions().get((symbol, exchange_id), {"quantity": 0, "avg_price": 0})
        
        # 4. QUANT-ONLY DECISION. No LLM in the loop: the regime-weighted composite of the
        # validated quant edges is the sole, authoritative signal. Deterministic = zero network
        # latency, zero 429/402 failures. Thresholds come from the backtested strategy config.
        composite = self.strategy_ensemble.composite_score(quant_signals, regime)
        buy_th = self.cfg.get("buy_threshold", 0.58)
        sell_th = self.cfg.get("sell_threshold", 0.42)

        # SHADOW: [RL_METRICS] per-symbol shadow logging, same non-blocking additive style as
        # [SHADOW-ANOMALY]/[SHADOW-MEMORY] above. Reuses current_pos (already fetched for position
        # mgmt) and composite (already computed) -- no new API calls, never read decision-affecting.
        try:
            _qty = float(current_pos.get("quantity", 0) or 0)
            _avg = float(current_pos.get("avg_price", 0) or 0)
            reward = (float(price) - _avg) * _qty if _avg else 0.0          # unrealized pnl on open position
            exposure = abs(_qty) * float(price)
            uncertainty = 1.0 - min(abs(composite - 0.5) * 2.0, 1.0)        # closer to decision boundary = riskier
            penalty = exposure * uncertainty
            net_score = reward - penalty
            log.debug(f"[RL_METRICS] agent=supervisor symbol={symbol} reward=${reward:+.2f} "
                      f"penalty=${penalty:+.2f} net_score=${net_score:+.2f} composite={composite:.2f}")
        except Exception as e:
            log.debug(f"[RL_METRICS] skipped: {e}")

        # Trend filter (mirrors backtester.simulate): only take mean-reversion entries that are
        # ALIGNED with the macro trend -- buy dips in an uptrend, sell rips in a downtrend.
        # Fighting the trend (shorting a bull market) was the dominant loss source in backtest.
        import numpy as _np
        sma = float(_np.mean(price_list)) if price_list else float(price)
        uptrend = float(price) >= sma

        # Per-symbol whitelist: only OPEN new positions on symbols the backtester validated
        # (positive edge, beats hold). Empty whitelist -> trade nothing new. Non-whitelisted
        # symbols are still monitored and their existing positions still exit via settlement.
        whitelist = self.cfg.get("symbol_whitelist", [])
        symbol_ok = (symbol in whitelist) if whitelist else False

        if composite > buy_th and uptrend and symbol_ok:
            det_action = "BUY"
        elif composite < sell_th and not uptrend and symbol_ok:
            det_action = "SELL"   # opens a SHORT in a downtrend -> hedge engages when trend is down
        elif composite > 0.70 and symbol_ok:
            # Strong-signal override: composite is decisively bullish even if below SMA.
            # Bypasses uptrend filter to avoid missing entries on strong mean-reversion setups.
            det_action = "BUY"
        elif composite < 0.30 and symbol_ok:
            # Strong-signal override: composite is decisively bearish.
            det_action = "SELL"
        else:
            det_action = "HOLD"

        signal = {"action": det_action, "reason": f"Quant composite {composite:.2f} in {regime}"}

        # 5. Meta-Learner Judge (sizes confidence around the quant action). NOTE: the feature key
        # "llm_signal" is a LEGACY column name the trained XGBoost model expects -- it is fed the
        # QUANT direction (quant_dir), NOT any LLM output. Renaming it would break the model's
        # feature schema; the value is 100% quant-derived.
        quant_dir = 1.0 if det_action == "BUY" else (-1.0 if det_action == "SELL" else 0.0)
        meta_features = {**quant_signals, "llm_signal": quant_dir}
        meta_confidence = self.meta_learner.predict_confidence(meta_features)
        # OVERRIDE: if composite signal is strong but meta-learner is miscalibrated,
        # allow trade at minimum floor so entries aren't permanently suppressed.
        # Only fires on genuinely strong signals (composite > 0.65 or < 0.35).
        min_conf_floor = 0.25
        strong_bullish = composite > 0.65
        strong_bearish = composite < 0.35
        if meta_confidence < min_conf_floor and (strong_bullish or strong_bearish):
            log.info(f"[{exchange_id}] Meta-learner override: {meta_confidence:.2f} -> {min_conf_floor} (composite={composite:.2f} is strong)")
            meta_confidence = min_conf_floor
        signal["confidence"] = meta_confidence

        # SHADOW: case-based recall from vector memory. Purely observational -- logged for
        # comparison against the quant decision, never read by anything decision-affecting.
        mem_context = {
            "current_price": float(price),
            "indicators": {"rsi_14": quant_signals.get("momentum", 0.5) * 100, "trend": trend_1h},
            "market_regime": regime,
        }
        try:
            similar = self.vector_memory.retrieve_similar_decisions(symbol, mem_context)
            if similar:
                agree = sum(1 for m in similar if m.get("action") == det_action)
                log.debug(f"[SHADOW-MEMORY] {symbol}: {agree}/{len(similar)} similar past decisions agree with quant action {det_action}")
        except Exception as e:
            log.debug(f"[SHADOW-MEMORY] recall skipped: {e}")

        min_conf = 0.25  # Safety valve: meta-learner was suppressing all entries at 0.45
        if signal["action"] == "HOLD" or signal["confidence"] < min_conf:
            log.info(f"[{exchange_id}] Decision: HOLD {symbol} ({signal['confidence']:.2f}) | composite={composite:.2f} | regime={regime}")
            try:
                self.vector_memory.store_decision(symbol, mem_context, signal)
            except Exception as e:
                log.debug(f"[SHADOW-MEMORY] store skipped: {e}")
            return signal

        # 6. Risk & Execution
        # Compute Sortino of recent price path for risk sizing
        sortino = 1.0
        if len(price_list) >= 10:
            returns = pd.Series(price_list).pct_change().dropna()
            if not returns.empty:
                downside_returns = returns[returns < 0.0]
                downside_deviation = _np.sqrt(_np.mean(downside_returns ** 2)) if len(downside_returns) > 0 else 1e-9
                sortino = float(returns.mean() / downside_deviation) if downside_deviation > 0 else 1.0

        risk_evaluation = self.risk_agent.evaluate_trade(symbol, signal["action"], signal["confidence"], float(price), regime=regime, sortino=sortino)
        if not risk_evaluation.get("approved", False):
            log.warning(f"[{exchange_id}] Trade REJECTED by Risk Agent for {symbol}: {risk_evaluation.get('reason', 'Unknown reason')}")
            rejected = {"action": "HOLD", "confidence": signal["confidence"], "reason": risk_evaluation.get("reason")}
            try:
                self.vector_memory.store_decision(symbol, mem_context, rejected)
            except Exception as e:
                log.debug(f"[SHADOW-MEMORY] store skipped: {e}")
            return rejected

        risk_evaluation["target_exchange"] = exchange_id
        risk_evaluation["metadata"] = {
            "quant_signals": quant_signals,
            "meta_confidence": signal["confidence"],
            "quant_action": signal["action"],   # quant-derived; no LLM in the decision path
            "regime": regime,
            "trend_1h": trend_1h,
            "trend_4h": trend_4h
        }
        
        log.info(f"[{exchange_id}] EXECUTING VERIFIED TRADE: {signal['action']} for {symbol}")
        result = await self.execution_agent.execute_trade(risk_evaluation)
        try:
            self.vector_memory.store_decision(symbol, mem_context, signal)
        except Exception as e:
            log.debug(f"[SHADOW-MEMORY] store skipped: {e}")
        return {"action": signal["action"], "status": result['status']}
