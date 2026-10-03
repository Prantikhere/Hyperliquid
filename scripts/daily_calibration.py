#!/usr/bin/env python3
"""
Daily Calibration & Fine-Tuning Engine
Executes after every day's run to:
1. Collect & analyze 24h trading performance (DB & trade lessons)
2. Execute parameter tuning (buy/sell thresholds, min_confidence)
3. Retrain XGBoost Meta-Learner if new trade outcomes exist
4. Audit Strategy Lifecycle and Regime Performance
5. Update models_local/strategy_config.json if calibration criteria met
"""
import os
import sys
import json
import time
from datetime import datetime, timezone, timedelta

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import redis
from src.utils.db import DatabaseManager
from src.utils.logger import log
from src.autonomous.tuner import ParameterTuner
from src.autonomous.lifecycle import StrategyLifecycle
from src.intelligence.train_meta import train_meta_learner
from src.intelligence.train_laya import train_and_recalibrate_laya

def run_daily_calibration(force: bool = False):
    print("=" * 60)
    print(f"DAILY CALIBRATION & FINE-TUNING RUN — {datetime.now(timezone.utc).isoformat()} UTC")
    print("=" * 60)

    db = DatabaseManager()
    r = redis.Redis(host=os.getenv('REDIS_HOST', 'localhost'), port=6379, decode_responses=True)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "actions_taken": [],
        "metrics": {},
        "adjustments": {}
    }

    # -------------------------------------------------------------
    # 1. 24-Hour Performance Data Collection
    # -------------------------------------------------------------
    print("\n[1/5] Collecting 24h trade performance...")
    query = """
    SELECT market_id, side, price, size, status, metadata, time
    FROM system_trades
    WHERE time >= NOW() - INTERVAL '24 hours'
    ORDER BY time ASC
    """
    trades_24h = db.execute_query(query) or []
    report["metrics"]["total_trades_24h"] = len(trades_24h)
    print(f"  Total trade records logged in last 24h: {len(trades_24h)}")

    completed_trades = [t for t in trades_24h if t[4] in ('CLOSED', 'SETTLED', 'LIVE_OK') and t[5] and 'outcome' in t[5]]
    report["metrics"]["completed_trades_24h"] = len(completed_trades)
    print(f"  Completed trades with outcomes in last 24h: {len(completed_trades)}")

    # -------------------------------------------------------------
    # 2. Parameter Tuning Check (buy/sell thresholds, confidence)
    # -------------------------------------------------------------
    print("\n[2/5] Evaluating Parameter Calibration (ParameterTuner)...")
    config_path = os.path.join(PROJECT_ROOT, "models_local", "strategy_config.json")
    tuner = ParameterTuner(db=db, redis_client=r, config_path=config_path)
    
    perf = tuner.analyze_performance()
    report["metrics"]["rolling_performance"] = perf
    print(f"  7-Day Rolling Metrics: {perf['trades']} trades | Win Rate: {perf['win_rate']:.1%} | Avg ROI: {perf['avg_roi']:.2f}% | Sharpe: {perf['sharpe']:.2f} | Max DD: {perf.get('drawdown', 0):.2f}%")

    # Run tuner (force=True if daily calibration trigger)
    adjustments = tuner.tune(force=force or (perf["trades"] >= tuner.MIN_TRADES_FOR_TUNING))
    if adjustments:
        print(f"  ✅ PARAMETER ADJUSTMENTS APPLIED: {adjustments}")
        report["adjustments"]["parameters"] = adjustments
        report["actions_taken"].append(f"Adjusted trading parameters: {list(adjustments.keys())}")
    else:
        print("  ℹ️ Parameters remain optimal; no threshold adjustments needed at this time.")
        report["adjustments"]["parameters"] = "OPTIMAL_NO_CHANGE"

    # -------------------------------------------------------------
    # 3. Strategy Lifecycle & Decay
    # -------------------------------------------------------------
    print("\n[3/5] Applying Strategy Lifecycle Calibration...")
    lifecycle = StrategyLifecycle(db=db, redis_client=r)
    lifecycle.apply_daily_decay()
    retirement = lifecycle.get_retirement_candidates()
    if retirement:
        print(f"  ⚠️ Strategy Retirement Candidates detected: {retirement}")
        report["actions_taken"].append(f"Identified retirement candidates: {retirement}")
    else:
        print("  ✅ Strategy lifecycle weights decayed normally; 0 retirement candidates.")

    # -------------------------------------------------------------
    # 4. Meta-Learner Retraining Check
    # -------------------------------------------------------------
    print("\n[4/5] Checking Meta-Learner Retraining...")
    ml_query = """
    SELECT COUNT(*) FROM system_trades 
    WHERE status = 'LIVE_OK' 
    AND (metadata->>'outcome') IS NOT NULL
    AND time >= NOW() - INTERVAL '30 days'
    """
    ml_samples = db.execute_query(ml_query)
    sample_count = ml_samples[0][0] if ml_samples else 0
    print(f"  Available labeled samples for Meta-Learner: {sample_count}")
    
    if sample_count >= 10:
        print("  🔄 Retraining XGBoost Meta-Learner with latest labeled trade data...")
        try:
            train_meta_learner()
            print("  ✅ Meta-Learner retraining completed successfully.")
            report["actions_taken"].append("Retrained XGBoost Meta-Learner")
        except Exception as e:
            print(f"  ❌ Meta-Learner training error: {e}")
            report["actions_taken"].append(f"Meta-Learner training failed: {e}")
    else:
        print(f"  ℹ️ Insufficient new trade samples ({sample_count}/10) for retraining. Skipping model retrain.")

    # -------------------------------------------------------------
    # 5. Dynamic Whitelist Audit
    # -------------------------------------------------------------
    print("\n[5/6] Auditing Universe & Dynamic Whitelist...")
    cached_wl = r.get("dynamic_whitelist:symbols:hyperliquid")
    wl_symbols = json.loads(cached_wl) if cached_wl else []
    print(f"  Active Dynamic Whitelist: {len(wl_symbols)} symbols")

    # -------------------------------------------------------------
    # 6. Laya Continuous RL & Recalibration Engine
    # -------------------------------------------------------------
    print("\n[6/6] Executing Laya Continuous RL & Decision Recalibration...")
    try:
        laya_res = train_and_recalibrate_laya(force=force)
        if laya_res.get("status") == "success":
            print(f"  ✅ Laya Recalibrated: ECE reduced by {laya_res.get('ece_reduction_pct')}% | Admitted Win Rate: {laya_res.get('admitted_win_rate'):.1%}")
            report["actions_taken"].append(f"Recalibrated Laya Decision Engine (ECE improvement {laya_res.get('ece_reduction_pct')}%)")
            report["metrics"]["laya_calibration"] = laya_res
        else:
            print(f"  ℹ️ Laya calibration skipped: {laya_res.get('reason')}")
    except Exception as e:
        print(f"  ❌ Laya calibration error: {e}")
        report["actions_taken"].append(f"Laya calibration failed: {e}")

    # Record calibration history
    calib_log_path = os.path.join(PROJECT_ROOT, "logs", "calibration.log")
    os.makedirs(os.path.dirname(calib_log_path), exist_ok=True)
    with open(calib_log_path, "a") as f:
        f.write(json.dumps(report) + "\n")
    print(f"\n✅ Calibration cycle finished. Summary appended to {calib_log_path}")
    print("=" * 60)
    return report

if __name__ == "__main__":
    force_run = "--force" in sys.argv
    run_daily_calibration(force=force_run)
