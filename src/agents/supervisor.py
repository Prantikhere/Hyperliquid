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
from src.intelligence.laya_client import (
    get_laya_client, LAYA_ENTRY_QUESTIONS, interpret_entry_verdict, LAYA_VETO_PROB,
)
from src.quant.anomaly_detector import AnomalyDetector
from src.quant.learning_module import learning_module
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
        self.learning_module = learning_module  # Expose for brain feedback loop
        self.laya_client = get_laya_client()  # System-1 cross-check: veto/adjust only
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

    def _get_whitelist(self, exchange_id='hyperliquid'):
        """Get whitelist using dynamic scoring with static fallback."""
        # Check if dynamic whitelist is enabled
        if not self.cfg.get('use_dynamic_whitelist', True):
            log.debug("[DYN_WHITELIST] Dynamic whitelist disabled, using static")
            return self.cfg.get("symbol_whitelist", [])
        
        try:
            from src.quant.dynamic_whitelist import DynamicWhitelist
            top_n = self.cfg.get('dynamic_whitelist_top_n', 8)
            update_hours = self.cfg.get('dynamic_whitelist_update_hours', 4)
            
            dyn_wl = DynamicWhitelist(
                exchange_id=exchange_id,
                top_n=top_n,
                update_interval_hours=update_hours
            )
            whitelist = dyn_wl.get_whitelist()
            if whitelist:
                log.debug(f"[DYN_WHITELIST] Active whitelist ({len(whitelist)} symbols): {whitelist}")
                return whitelist
        except Exception as e:
            log.warning(f"[DYN_WHITELIST] Dynamic scoring failed: {e}, using static fallback")
        
        # Fallback to static whitelist from config
        return self.cfg.get("symbol_whitelist", [])

    async def run_cycle(self, symbol, exchange_id="bingx"):
        log.info(f"--- [{exchange_id.upper()}] Full-Potential Cycle: {symbol} ---")
        
        # Dynamic Bankroll Syncing
        try:
            balance = await self.execution_agent.multi_client.get_balance(exchange_id)
            if balance:
                if exchange_id == 'hyperliquid':
                    # SDK returns: {'account_value': float, 'margin_used': float, 'available': float}
                    equity = float(balance.get('account_value', 0))
                else:
                    equity = float(balance.get('total', {}).get('USDT', 0) or balance.get('total', {}).get('USDC', 0) or 0)
                
                if equity > 0:
                    self.risk_agent.risk_manager.update_bankroll(equity)
                    # TRACK PEAK VALUE FOR DRAWDOWN CALCULATION
                    try:
                        peak_key = "account_peak_value"
                        current_key = "account_current_value"
                        current_peak = float(self.redis.get(peak_key) or 0)
                        if equity > current_peak:
                            self.redis.set(peak_key, equity, ex=2592000)  # 30 day TTL
                        self.redis.set(current_key, equity, ex=2592000)
                    except Exception:
                        pass
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

        # BLOCK UNKNOWN REGIME: 54 trades, 0 wins, -11.18% ROI
        if regime == "UNKNOWN":
            log.info(f"[REGIME_BLOCK] {symbol}: UNKNOWN regime blocked (0% historical win rate)")
            return {"action": "HOLD", "confidence": 0, "reason": "UNKNOWN regime blocked"}

        # AUTONOMOUS BRAIN: Get recommendations before decision logic
        brain_recommendation = None
        if hasattr(self, 'brain') and self.brain:
            try:
                brain_recommendation = self.brain.get_trade_recommendation(symbol, regime)
                if not brain_recommendation.get("should_trade", True):
                    log.info(f"[BRAIN] {symbol} blocked: {brain_recommendation.get('reason', 'unknown')}")
                    return {"action": "HOLD", "confidence": 0, "reason": f"Brain blocked: {brain_recommendation.get('reason')}"}
                # Apply confidence adjustment from brain
                if brain_recommendation.get("confidence_adjustment", 1.0) != 1.0:
                    log.debug(f"[BRAIN] {symbol} confidence adjustment: {brain_recommendation['confidence_adjustment']:.2f}")
            except Exception as e:
                log.error(f"[BRAIN] Recommendation failed: {e}")

        # SHADOW: structural-break anomaly score, independent of the rule-based DD-kill.
        # Never gates or sizes anything -- logged only, for future evaluation.
        # Use cached ER/vol_ratio from RegimeEngine (avoids duplicate computation)
        try:
            er = self.regime_engine.last_er if self.regime_engine.last_er is not None else 0.5
            vol_ratio = self.regime_engine.last_vol_ratio if self.regime_engine.last_vol_ratio is not None else 1.0
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
        
        # Use brain-tuned parameters if available
        if brain_recommendation and brain_recommendation.get("tuner_params"):
            tuner = brain_recommendation["tuner_params"]
            buy_th = tuner.get("buy_threshold", self.cfg.get("buy_threshold", 0.58))
            sell_th = tuner.get("sell_threshold", self.cfg.get("sell_threshold", 0.42))
        else:
            buy_th = self.cfg.get("buy_threshold", 0.58)
            sell_th = self.cfg.get("sell_threshold", 0.42)

        # SHADOW: [EDGE_METRICS] per-symbol shadow logging. Tracks unrealized P&L and
        # signal uncertainty — purely observational, never decision-affecting.
        try:
            _qty = float(current_pos.get("quantity", 0) or 0)
            _avg = float(current_pos.get("avg_price", 0) or 0)
            reward = (float(price) - _avg) * _qty if _avg else 0.0          # unrealized pnl on open position
            exposure = abs(_qty) * float(price)
            uncertainty = 1.0 - min(abs(composite - 0.5) * 2.0, 1.0)        # closer to decision boundary = riskier
            penalty = exposure * uncertainty
            net_score = reward - penalty
            log.debug(f"[EDGE_METRICS] agent=supervisor symbol={symbol} reward=${reward:+.2f} "
                      f"penalty=${penalty:+.2f} net_score=${net_score:+.2f} composite={composite:.2f}")
        except Exception as e:
            log.debug(f"[EDGE_METRICS] skipped: {e}")

        # Trend filter (mirrors backtester.simulate): only take mean-reversion entries that are
        # ALIGNED with the macro trend -- buy dips in an uptrend, sell rips in a downtrend.
        # Fighting the trend (shorting a bull market) was the dominant loss source in backtest.
        import numpy as _np
        sma = float(_np.mean(price_list)) if price_list else float(price)
        uptrend = float(price) >= sma

        # Per-symbol whitelist: DYNAMIC scoring based on volume, momentum, volatility
        # Falls back to static list if dynamic scoring fails
        whitelist = self._get_whitelist(exchange_id)
        symbol_ok = (symbol in whitelist) if whitelist else False
        
        # DEAD SYMBOL CHECK: Skip symbols with broken oracle/exchange feeds
        dead_key = f"dead_symbols:{exchange_id}"
        dead_symbols = self.redis.smembers(dead_key)
        if symbol in dead_symbols:
            log.warning(f"[DEAD_SYMBOL] {symbol} on {exchange_id} is blacklisted. Skipping.")
            return {"action": "HOLD", "confidence": 0, "reason": f"Dead symbol: {symbol}"}

        # LAYA System-1 cross-check: cached/background verdict (zero added latency).
        # Used ONLY to veto or scale the quant decision -- never to invent entries
        # (quant-only decision policy preserved). First cycle warms the cache.
        laya_verdict = None
        laya_interp = None
        try:
            if symbol_ok:
                laya_verdict = self.laya_client.evaluate(
                    self._build_laya_state(symbol, exchange_id, price, price_list, regime,
                                           er, vol_ratio, trend_1h, trend_4h,
                                           quant_signals, composite, current_pos),
                    LAYA_ENTRY_QUESTIONS,
                    cache_key=f"entry:{symbol}:{regime}",
                )
                if laya_verdict:
                    laya_interp = interpret_entry_verdict(laya_verdict, engine_regime=regime)
        except Exception as e:
            log.debug(f"[LAYA] evaluate skipped: {e}")

        if composite > buy_th and uptrend and symbol_ok:
            det_action = "BUY"
        elif composite < sell_th and not uptrend and symbol_ok:
            det_action = "SELL"   # opens a SHORT in a downtrend -> hedge engages when trend is down
        elif composite >= 0.65 and symbol_ok and ("MEAN_REVERTING" in regime.upper() or "NEUTRAL" in regime.upper()):
            # Strong-signal override: composite is decisively bullish in MEAN_REVERTING/NEUTRAL regime.
            # Raised from 0.60 to 0.65 to require higher conviction for entries.
            det_action = "BUY"
            log.debug(f"[ENTRY_DEBUG] {symbol}: Strong-signal BUY: composite={composite:.2f}, regime={regime}, symbol_ok={symbol_ok}")
        elif composite <= 0.35 and symbol_ok and ("MEAN_REVERTING" in regime.upper() or "NEUTRAL" in regime.upper()):
            # Strong-signal override: composite is decisively bearish in MEAN_REVERTING/NEUTRAL regime.
            # Raised from 0.40 to 0.35 to require higher conviction for entries.
            det_action = "SELL"
            log.debug(f"[ENTRY_DEBUG] {symbol}: Strong-signal SELL: composite={composite:.2f}, regime={regime}, symbol_ok={symbol_ok}")
        else:
            det_action = "HOLD"
            log.debug(f"[ENTRY_DEBUG] {symbol}: HOLD: composite={composite:.2f}, regime={regime}, symbol_ok={symbol_ok}, buy_th={buy_th}, uptrend={uptrend}")

        # LAYA VETO: strong System-1 "avoid" blocks a quant entry before it is sized.
        # Veto-only by design -- laya can never create a BUY/SELL on its own.
        if det_action in ("BUY", "SELL") and laya_interp and laya_interp["veto"]:
            log.info(f"[LAYA_VETO] {symbol}: {det_action} blocked in {regime} "
                     f"(avoid_p={laya_interp['avoid_prob']:.2f}, noul={laya_interp['noul']:.2f}, "
                     f"composite={composite:.2f})")
            return {"action": "HOLD", "confidence": 0,
                    "reason": f"Laya veto (avoid_p={laya_interp['avoid_prob']:.2f})"}

        signal = {"action": det_action, "reason": f"Quant composite {composite:.2f} in {regime}"}

        # CHURN GUARD: Check if this symbol is in cooldown after a recent exit
        # Prevents the enter->stale exit->immediate re-enter loop
        import time as _time
        cooldown_key = f"entry_cooldown:{exchange_id}:{symbol}"
        cooldown_until = self.redis.get(cooldown_key)
        if cooldown_until:
            remaining = float(cooldown_until) - _time.time()
            if remaining > 0:
                log.info(f"[CHURN_GUARD] {symbol} on {exchange_id}: in cooldown for {remaining:.0f}s more. Blocking entry.")
                signal = {"action": "HOLD", "confidence": 0, "reason": f"Cooldown active ({remaining:.0f}s remaining)"}
                return signal

        # SESSION LOSS GUARD: Check if we've lost too much today
        session_loss_key = f"session_realized_pnl:{_time.time() // 86400}"
        session_loss = float(self.redis.get(session_loss_key) or 0)
        if session_loss < -5.0:  # More than $5 lost today
            log.warning(f"[SESSION_GUARD] Daily realized loss ${session_loss:.2f} exceeds $5 limit. Blocking new entries.")
            signal = {"action": "HOLD", "confidence": 0, "reason": f"Session loss limit breached: ${session_loss:.2f}"}
            return signal

        # CHURN GUARD: Check exit count for this symbol today
        exit_count_key = f"exit_count:{exchange_id}:{symbol}:{int(_time.time() / 86400)}"
        exit_count = int(self.redis.get(exit_count_key) or 0)
        if exit_count >= 4:
            log.warning(f"[CHURN_GUARD] {symbol} on {exchange_id}: {exit_count} exits today. Blocking re-entry.")
            signal = {"action": "HOLD", "confidence": 0, "reason": f"Churn limit breached ({exit_count} exits today)"}
            return signal

        # 5. Meta-Learner Judge (sizes confidence around the quant action). NOTE: the feature key
        # "llm_signal" is a LEGACY column name the trained XGBoost model expects -- it is fed the
        # QUANT direction (quant_dir), NOT any LLM output. Renaming it would break the model's
        # feature schema; the value is 100% quant-derived.
        quant_dir = 1.0 if det_action == "BUY" else (-1.0 if det_action == "SELL" else 0.0)
        meta_features = {**quant_signals, "llm_signal": quant_dir}
        # Pass pre-computed RNN result from StrategyEnsemble to avoid double RNN call
        meta_confidence = self.meta_learner.predict_confidence(meta_features, rnn_result=self.strategy_ensemble.last_rnn_result)
        # OVERRIDE: if composite signal is strong but meta-learner is miscalibrated,
        # allow trade at minimum floor so entries aren't permanently suppressed.
        # Only fires when the quant action is BUY/SELL (not HOLD) AND composite is strong.
        min_conf_floor = 0.40
        actionable = det_action in ("BUY", "SELL")
        strong_signal = abs(composite - 0.5) >= 0.20  # composite >= 0.70 or <= 0.30
        if meta_confidence < min_conf_floor and actionable and strong_signal:
            log.info(f"[{exchange_id}] Meta-learner override: {meta_confidence:.2f} -> {min_conf_floor} (composite={composite:.2f}, action={det_action})")
            meta_confidence = min_conf_floor
        signal["confidence"] = meta_confidence

        # LAYA conviction/regime blend: bounded multiplier on meta-confidence.
        # Range ~[0.85, 1.08] -- can soften a weak setup or sharpen a coherent one,
        # but the quant action itself is never flipped here (veto path handles blocks).
        if laya_interp and actionable:
            before = signal["confidence"]
            signal["confidence"] = max(0.0, min(1.0, before * laya_interp["multiplier"]))
            if laya_interp["regime_mismatch"]:
                log.info(f"[LAYA_REGIME] {symbol}: engine={regime} vs laya={laya_interp['regime']} "
                         f"(p={laya_interp['regime_prob']:.2f}) -- confidence scaled "
                         f"{before:.3f} -> {signal['confidence']:.3f}")
            elif abs(signal["confidence"] - before) > 0.005:
                log.debug(f"[LAYA] {symbol} confidence {before:.3f} -> "
                          f"{signal['confidence']:.3f} ({laya_interp['summary']})")

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

        min_conf = 0.40  # Must match min_conf_floor used in override logic
        log.debug(f"[DECISION_DEBUG] {symbol}: det_action={det_action}, signal_action={signal.get('action')}, confidence={signal.get('confidence', 0):.2f}, composite={composite:.2f}")
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

        risk_evaluation = self.risk_agent.evaluate_trade(symbol, signal["action"], signal["confidence"], float(price), regime=regime, sortino=sortino, exchange_id=exchange_id)
        if not risk_evaluation.get("approved", False):
            log.warning(f"[{exchange_id}] Trade REJECTED by Risk Agent for {symbol}: {risk_evaluation.get('reason', 'Unknown reason')}")
            rejected = {"action": "HOLD", "confidence": signal["confidence"], "reason": risk_evaluation.get("reason")}
            try:
                self.vector_memory.store_decision(symbol, mem_context, rejected)
            except Exception as e:
                log.debug(f"[SHADOW-MEMORY] store skipped: {e}")
            return rejected

        # PRODUCTION MONITORING: Track P&L and alert on critical events
        self._track_production_metrics(equity, risk_evaluation, symbol)

        risk_evaluation["target_exchange"] = exchange_id
        risk_evaluation["metadata"] = {
            "quant_signals": quant_signals,
            "meta_confidence": signal["confidence"],
            "quant_action": signal["action"],   # quant-derived; no LLM in the decision path
            "regime": regime,
            "trend_1h": trend_1h,
            "trend_4h": trend_4h,
            # Laya System-1 verdict attached for post-trade forensics + /feedback loop.
            "laya": ({
                "summary": laya_interp["summary"],
                "entry": laya_interp["entry"],
                "avoid_prob": round(laya_interp["avoid_prob"], 4),
                "regime": laya_interp["regime"],
                "decision_id": (laya_verdict or {}).get("decision_id"),
            } if laya_interp else None),
        }
        
        log.info(f"[{exchange_id}] EXECUTING VERIFIED TRADE: {signal['action']} for {symbol}")
        result = await self.execution_agent.execute_trade(risk_evaluation)
        
        # Record entry time for stale exit tracking
        if result.get("status") == "OK":
            import time as _entry_time
            self.redis.set(f"position_age:{exchange_id}:{symbol}", _entry_time.time(), ex=86400)
            # Track entry count for churn detection
            entry_count_key = f"entry_count:{exchange_id}:{symbol}:{int(_entry_time.time() / 86400)}"
            current_entry_count = int(self.redis.get(entry_count_key) or 0)
            self.redis.set(entry_count_key, current_entry_count + 1, ex=172800)
            # Stash the Laya entry verdict (incl. knowledge-store decision_id) so
            # settlement/forensics can send /feedback when the trade closes.
            if laya_interp:
                try:
                    self.redis.set(
                        f"laya_entry:{exchange_id}:{symbol}",
                        json.dumps({
                            "entry": laya_interp["entry"],
                            "avoid_prob": round(laya_interp["avoid_prob"], 4),
                            "regime": regime,
                            "decision_id": (laya_verdict or {}).get("decision_id"),
                        }),
                        ex=604800,  # 7 days
                    )
                except Exception as e:
                    log.debug(f"[LAYA] entry stash skipped: {e}")
        
        try:
            self.vector_memory.store_decision(symbol, mem_context, signal)
        except Exception as e:
            log.debug(f"[SHADOW-MEMORY] store skipped: {e}")
        return {"action": signal["action"], "status": result['status']}

    def _build_laya_state(self, symbol, exchange_id, price, price_list, regime,
                          er, vol_ratio, trend_1h, trend_4h, quant_signals,
                          composite, current_pos) -> dict:
        """Compact quant-feature summary as the laya state payload."""
        try:
            sig_parts = []
            for k, v in (quant_signals or {}).items():
                try:
                    sig_parts.append(f"{k}={float(v):.3f}")
                except (TypeError, ValueError):
                    pass
            rnn = self.strategy_ensemble.last_rnn_result or {}
            qty = float(current_pos.get("quantity", 0) or 0)
            pos_desc = ("open qty=%.4f" % qty) if qty else "flat"
            recent = [round(float(p), 6) for p in (price_list or [])[-5:]]
            body = (
                f"{symbol} on {exchange_id}. price={price}, regime={regime}, "
                f"composite={composite:.3f}, er={er:.2f}, vol_ratio={vol_ratio:.2f}, "
                f"trend_1h={trend_1h}, trend_4h={trend_4h}, "
                f"signals[{', '.join(sig_parts)}], "
                f"rnn={rnn.get('signal', 'NA')}({float(rnn.get('confidence', 0)):.2f}), "
                f"position={pos_desc}, recent_prices={recent}"
            )
            return {"symbol": symbol, "body": body}
        except Exception as e:
            return {"symbol": symbol, "body": f"{symbol} regime={regime} state unavailable ({e})"}

    def _track_production_metrics(self, equity, risk_evaluation, symbol):
        """Track production metrics and alert on critical events."""
        try:
            import redis
            import time
            
            r = redis.Redis(host=os.getenv('REDIS_HOST', 'localhost'), port=6379, decode_responses=True)
            
            # Store equity for drawdown tracking
            peak_key = "account_peak_value"
            current_key = "account_current_value"
            peak_value = float(r.get(peak_key) or 0)
            
            if equity > peak_value:
                r.set(peak_key, equity, ex=2592000)  # 30 day TTL
            r.set(current_key, equity, ex=2592000)
            
            # Calculate drawdown
            drawdown_pct = 0
            if peak_value > 0:
                drawdown_pct = ((peak_value - equity) / peak_value) * 100
            
            # Alert thresholds
            if drawdown_pct >= 10:
                log.warning(f"[PRODUCTION ALERT] Drawdown: {drawdown_pct:.1f}% (threshold: 10%)")
            if drawdown_pct >= 15:
                log.critical(f"[PRODUCTION CRITICAL] Max drawdown breached: {drawdown_pct:.1f}% - KILL SWITCH ACTIVE")
            
            # Track daily P&L
            daily_key = f"daily_pnl:{int(time.time() / 86400)}"
            daily_pnl = float(r.get(daily_key) or 0)
            
            # Log production status
            log.info(f"[PRODUCTION] Equity: ${equity:.2f} | Drawdown: {drawdown_pct:.1f}% | Daily P&L: ${daily_pnl:.2f}")
            
        except Exception as e:
            log.debug(f"Production metrics tracking error: {e}")
