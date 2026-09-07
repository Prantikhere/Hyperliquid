import os
import xgboost as xgb
import pandas as pd
from src.utils.logger import log

class MLModel:
    def __init__(self, model_path="models/xgboost_v1.json"):
        self.model_path = model_path
        self.model = None
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = xgb.Booster()
            self.model.load_model(self.model_path)
            log.info(f"Loaded ML model from {self.model_path}")
        else:
            log.warning("ML model file not found. System will rely on statistical signals.")

    def predict(self, features):
        if self.model is None:
            return 0.5 # Neutral
        
        # Convert features to DMatrix
        df = pd.DataFrame([features])
        dmatrix = xgb.DMatrix(df)
        prediction = self.model.predict(dmatrix)
        return float(prediction[0])

    def train(self, df, target_col="outcome"):
        """Train or retrain the model."""
        # Force numeric types to avoid XGBoost errors with 'object' dtypes (e.g. from Postgres Decimal)
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna()
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        log.info(f"Training on {len(X)} samples with features: {X.columns.tolist()}")
        dtrain = xgb.DMatrix(X, label=y)
        params = {
            'max_depth': 3,
            'eta': 0.1,
            'objective': 'binary:logistic',
            'eval_metric': 'auc'
        }
        self.model = xgb.train(params, dtrain, num_boost_round=100)
        self.model.save_model(self.model_path)
        log.info(f"Trained and saved ML model to {self.model_path}")
