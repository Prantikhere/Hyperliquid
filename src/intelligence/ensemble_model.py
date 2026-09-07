import xgboost as xgb
import numpy as np
import pandas as pd
from src.utils.logger import log

def _roi_to_confidence(raw_roi):
    """Convert raw ROI prediction (continuous, e.g. -0.05 to +0.10) to a
    confidence score (0-1). Uses sigmoid to center around 0 ROI = 0.5 confidence.
    The steepness parameter controls how aggressively ROI maps to confidence."""
    # Center at 0% ROI = 0.5 confidence, steepness=15 gives good spread:
    #   -5% ROI -> 0.18, 0% -> 0.50, +5% -> 0.82, +10% -> 0.95
    return 1.0 / (1.0 + np.exp(-15.0 * raw_roi))

def _platt_scale(raw_prob, a=0.8, b=1.0):
    """Legacy Platt scaling for old binary classifier models.
    Kept for backward compatibility with existing trained models."""
    p = np.clip(raw_prob, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1.0 - p))
    return 1.0 / (1.0 + np.exp(-(a * logit + b)))

class EnsembleMetaLearner:
    """
    XGBoost Meta-Learner that combines multiple signal sources.
    Supports both:
      - Legacy binary classifier (objective='binary:logistic') with Platt scaling
      - New regressor (objective='reg:squarederror') with sigmoid ROI-to-confidence
    """
    def __init__(self, model_path="models_local/meta_learner.json"):
        self.model_path = model_path
        self.model = xgb.Booster()
        self.is_regressor = False
        try:
            self.model.load_model(model_path)
            self.is_trained = True
            # Detect model type from saved feature names (regressor has 'trend')
            feature_names = self.model.feature_names or []
            self.is_regressor = 'trend' in feature_names
            if self.is_regressor:
                log.info("Loaded meta-learner: XGBRegressor (continuous ROI)")
            else:
                log.info("Loaded meta-learner: XGBClassifier (binary, Platt-scaled)")
        except:
            log.warning("No pre-trained meta-learner found. Using heuristic weights.")
            self.is_trained = False

    def predict_confidence(self, features):
        """
        features: {mean_reversion, momentum, order_flow, trend, llm_signal}
        Returns: meta_confidence (0 to 1).
        """
        if not self.is_trained:
            # Fallback to dynamic weighted heuristic
            llm_val = abs(features.get('llm_signal', 0.0))
            llm_sign = features.get('llm_signal', 0.0)
            
            momentum = features.get('momentum', 0.5)
            mean_reversion = features.get('mean_reversion', 0.5)
            order_flow = features.get('order_flow', 0.5)
            
            if llm_sign > 0:
                quant_align = (momentum + mean_reversion + order_flow) / 3.0
            elif llm_sign < 0:
                quant_align = ((1.0 - momentum) + (1.0 - mean_reversion) + (1.0 - order_flow)) / 3.0
            else:
                quant_align = 0.5
                
            return (llm_val * 0.7) + (quant_align * 0.3)

        try:
            if self.is_regressor:
                # New model: continuous ROI regression with 5 features
                df_cols = ['mean_reversion', 'momentum', 'order_flow', 'trend', 'llm_signal']
                df = pd.DataFrame([features])[df_cols]
                dmatrix = xgb.DMatrix(df)
                raw_roi = float(self.model.predict(dmatrix)[0])
                confidence = _roi_to_confidence(raw_roi)
                log.debug(f"[META_LEARNER] raw_roi={raw_roi*100:.3f}% -> confidence={confidence:.4f}")
                return confidence
            else:
                # Legacy model: binary classification with Platt scaling
                df_cols = ['mean_reversion', 'momentum', 'order_flow', 'llm_signal']
                df = pd.DataFrame([features])[df_cols]
                dmatrix = xgb.DMatrix(df)
                raw_pred = float(self.model.predict(dmatrix)[0])
                calibrated = _platt_scale(raw_pred)
                log.debug(f"[META_LEARNER] raw={raw_pred:.4f} -> calibrated={calibrated:.4f}")
                return calibrated
        except Exception as e:
            log.error(f"Meta-Learner Inference Error: {e}")
            return 0.5
