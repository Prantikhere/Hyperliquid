import numpy as np
from collections import deque
from sklearn.ensemble import IsolationForest
from src.utils.logger import log


class AnomalyDetector:
    """
    SHADOW-ONLY structural-break detector. Independent of the rule-based DD-kill (which only
    reacts to realized equity loss). This scores how unusual the CURRENT market microstructure
    is vs. its own recent history, using an IsolationForest over per-cycle quant features
    (mean_reversion, momentum, order_flow, trend, efficiency_ratio, vol_ratio).

    Not wired into risk_agent or any decision path. Call score() every cycle, it logs a flag
    when the current feature vector is an outlier vs. the rolling window. Purely observational
    until proven useful over real data.
    """
    def __init__(self, window=200, min_samples=40, refit_every=20, contamination=0.05):
        self.window = window
        self.min_samples = min_samples
        self.refit_every = refit_every
        self.contamination = contamination
        self.buffer = deque(maxlen=window)
        self.model = None
        self._since_fit = 0

    def _refit(self):
        X = np.array(self.buffer)
        try:
            self.model = IsolationForest(
                n_estimators=100, contamination=self.contamination, random_state=42
            ).fit(X)
        except Exception as e:
            log.debug(f"[SHADOW-ANOMALY] refit skipped: {e}")

    def score(self, features: dict):
        """features: {mean_reversion, momentum, order_flow, trend, er, vol_ratio} (all floats).
        Returns (is_anomaly: bool|None, raw_score: float|None). None until enough history."""
        vec = [
            features.get("mean_reversion", 0.5),
            features.get("momentum", 0.5),
            features.get("order_flow", 0.5),
            features.get("trend", 0.5),
            features.get("er", 0.0),
            features.get("vol_ratio", 1.0),
        ]
        self.buffer.append(vec)
        self._since_fit += 1

        if len(self.buffer) < self.min_samples:
            return None, None

        if self.model is None or self._since_fit >= self.refit_every:
            self._refit()
            self._since_fit = 0

        if self.model is None:
            return None, None

        try:
            pred = self.model.predict([vec])[0]          # -1 = anomaly, 1 = normal
            raw = float(self.model.decision_function([vec])[0])  # lower = more anomalous
            return (pred == -1), raw
        except Exception as e:
            log.debug(f"[SHADOW-ANOMALY] score skipped: {e}")
            return None, None
