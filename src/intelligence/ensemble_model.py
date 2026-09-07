import xgboost as xgb
import numpy as np
import pandas as pd
from src.utils.logger import log

class EnsembleMetaLearner:
    """
    XGBoost Meta-Learner that combines multiple signal sources.
    As per Blueprint 2.0, this provides a directional accuracy boost.
    """
    def __init__(self, model_path="models_local/meta_learner.json"):
        self.model_path = model_path
        self.model = xgb.Booster()
        try:
            self.model.load_model(model_path)
            self.is_trained = True
        except:
            log.warning("No pre-trained meta-learner found. Using heuristic weights.")
            self.is_trained = False

    def predict_confidence(self, features):
        """
        features: {mean_reversion, momentum, order_flow, llm_signal}
        Returns: meta_confidence (0 to 1)
        """
        if not self.is_trained:
            # Fallback to dynamic weighted heuristic
            # We want to reward alignment between LLM and Quants
            llm_val = abs(features.get('llm_signal', 0.0))
            llm_sign = features.get('llm_signal', 0.0)
            
            momentum = features.get('momentum', 0.5)
            mean_reversion = features.get('mean_reversion', 0.5)
            order_flow = features.get('order_flow', 0.5)
            
            if llm_sign > 0:
                # Bullish alignment (higher values indicate bullishness)
                quant_align = (momentum + mean_reversion + order_flow) / 3.0
            elif llm_sign < 0:
                # Bearish alignment (lower values indicate bearishness/reversion/selling pressure)
                quant_align = ((1.0 - momentum) + (1.0 - mean_reversion) + (1.0 - order_flow)) / 3.0
            else:
                quant_align = 0.5
                
            return (llm_val * 0.7) + (quant_align * 0.3)

        try:
            # Convert dict to DataFrame with expected column order
            df_cols = ['mean_reversion', 'momentum', 'order_flow', 'llm_signal']
            df = pd.DataFrame([features])[df_cols]
            dmatrix = xgb.DMatrix(df)
            preds = self.model.predict(dmatrix)
            return float(preds[0])
        except Exception as e:
            log.error(f"Meta-Learner Inference Error: {e}")
            return 0.5
