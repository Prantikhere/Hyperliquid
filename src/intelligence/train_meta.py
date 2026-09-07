
import xgboost as xgb
import pandas as pd
import numpy as np
import json
import os
from src.utils.db import DatabaseManager
from src.utils.logger import log

def calculate_trade_sortino(db, symbol, side, entry_price, entry_time):
    """
    TensorTrade-inspired Reward/Risk calculator.
    Simulates the price path of the trade from external_prices and computes
    a Sortino ratio to use as sample weighting for the meta-learner.
    """
    query = """
    SELECT price FROM external_prices 
    WHERE symbol IN (%s, %s, %s) AND time >= %s AND time <= %s + INTERVAL '24 hours'
    ORDER BY time ASC
    """
    try:
        rows = db.execute_query(query, (f"bingx:{symbol}", f"hyperliquid:{symbol}", symbol, entry_time, entry_time))
        if not rows or len(rows) < 2:
            return 1.0
            
        prices = [float(r[0]) for r in rows]
        returns = []
        sl_limit = 0.025
        tp_limit = 0.05
        
        for p in prices[1:]:
            roi = (p - entry_price) / entry_price if side == 'BUY' else (entry_price - p) / entry_price
            returns.append(roi)
            if roi >= tp_limit or roi <= -sl_limit:
                break
                
        if not returns:
            return 0.1
            
        ret_series = pd.Series(returns)
        mean_ret = ret_series.mean()
        downside_returns = ret_series[ret_series < 0]
        
        if downside_returns.empty:
            return 3.0
            
        downside_deviation = np.sqrt(np.mean(downside_returns ** 2))
        if downside_deviation == 0:
            return 3.0
            
        sortino = mean_ret / downside_deviation
        return float(np.clip(1.0 + sortino, 0.1, 5.0))
    except Exception as e:
        log.warning(f"Sortino calc error for {symbol} at {entry_time}: {e}")
        return 1.0

def train_meta_learner():
    db = DatabaseManager()
    log.info("Starting Meta-Learner retraining (continuous ROI labels)...")
    
    query = """
    SELECT metadata, market_id, side, price, time FROM system_trades 
    WHERE status = 'LIVE_OK' 
    AND (metadata->>'outcome') IS NOT NULL
    AND time >= '2026-06-18'
    LIMIT 1000
    """
    rows = db.execute_query(query)
    
    if len(rows) < 10:
        log.info(f"Insufficient data for training ({len(rows)} samples). Need at least 10.")
        return

    data = []
    sample_weights = []
    for row in rows:
        meta, symbol, side, price, time = row
        q = meta.get('quant_signals', {})
        features = {
            'mean_reversion': q.get('mean_reversion', 0.5),
            'momentum': q.get('momentum', 0.5),
            'order_flow': q.get('order_flow', 0.5),
            'trend': q.get('trend', 0.5),
            # legacy 'llm_signal' column is fed the QUANT action direction
            'llm_signal': (lambda a: 1.0 if a == 'BUY' else (-1.0 if a == 'SELL' else 0.0))(
                meta.get('quant_action', meta.get('llm_action')))
        }
        # Continuous ROI label (NOT binary). Regressed directly.
        roi = float(meta.get('outcome', 0))
        features['label'] = roi
        data.append(features)
        
        # Sortino-based risk adjusted sample weighting
        weight = calculate_trade_sortino(db, symbol, side, float(price), time)
        sample_weights.append(weight)

    df = pd.DataFrame(data)
    X = df.drop('label', axis=1)
    y = df['label']

    # Train XGBoost as a REGRESSOR (continuous ROI) instead of classifier (binary win/loss).
    # This gives the model a meaningful gradient: "how profitable will this trade be?" vs
    # just "will it be positive?" — which is far more useful for confidence calibration.
    model = xgb.XGBRegressor(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        objective='reg:squarederror',
        subsample=0.8,
        colsample_bytree=0.8,
    )
    model.fit(X, y, sample_weight=np.array(sample_weights))

    # Save Model
    os.makedirs("models_local", exist_ok=True)
    model.save_model("models_local/meta_learner.json")
    
    # Save calibration stats for Platt scaling
    preds = model.predict(X)
    stats = {
        "n_samples": len(rows),
        "mean_roi": float(y.mean()),
        "std_roi": float(y.std()),
        "mean_pred": float(preds.mean()),
        "std_pred": float(preds.std()),
        "feature_importance": dict(zip(X.columns, model.feature_importances_.tolist())),
    }
    with open("models_local/meta_learner_stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    
    log.info(f"Meta-Learner retrained: {len(rows)} samples, "
             f"mean_roi={y.mean()*100:.2f}%, mean_pred={preds.mean()*100:.2f}%, "
             f"std_pred={preds.std()*100:.2f}%")

if __name__ == "__main__":
    train_meta_learner()

