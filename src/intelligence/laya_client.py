"""
Laya System-1 decision-engine client.

Laya (convaiinnovations/laya) is a fast non-autoregressive decision model served
locally (default http://127.0.0.1:8080/predict). It answers typed questions
(choice / score / noul) about a text "state" in a single forward pass.

Design constraints for the trading hot path:
  * Zero added latency: evaluate() never blocks. Fresh verdicts come from an
    in-memory cache; stale/missing ones are queued to a background worker.
  * Never raises: any failure returns None so callers fall back to quant-only.
  * Circuit breaker: after N consecutive failures the circuit opens for a
    cooldown window and evaluate() short-circuits without touching the network.
  * Quant-only policy: Laya is NOT an LLM, but callers may still only use it to
    VETO or ADJUST quant decisions -- never to invent entries on its own.

NOTE on response schema: the model's `confidence` fields ship uncalibrated
temperatures (see server warning), so we always use `probabilities[choice]`
for choice strength and `noul` for coherence -- never the bare `confidence`.
"""
import json
import os
import queue
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, Optional

from src.utils.logger import log

LAYA_URL = os.getenv("LAYA_URL", "http://127.0.0.1:8080/predict")
LAYA_ENABLED = os.getenv("LAYA_ENABLED", "true").lower() in ("1", "true", "yes")
LAYA_VETO_PROB = float(os.getenv("LAYA_VETO_PROB", "0.70"))
LAYA_TIMEOUT = float(os.getenv("LAYA_TIMEOUT", "30"))
LAYA_CACHE_TTL = float(os.getenv("LAYA_CACHE_TTL", "300"))
LAYA_CIRCUIT_AFTER = int(os.getenv("LAYA_CIRCUIT_AFTER", "3"))
LAYA_CIRCUIT_COOLDOWN = float(os.getenv("LAYA_CIRCUIT_COOLDOWN", "120"))

# Entry cross-check questions for the supervisor decision path.
LAYA_ENTRY_QUESTIONS = {
    "entry": {
        "type": "choice",
        "instructions": "Should the system enter this trade given the quant stats?",
        "criteria": {
            "enter": "strong quant edge, regime favorable, acceptable risk",
            "avoid": "weak edge, unfavorable regime, or elevated risk",
        },
    },
    "regime": {
        "type": "choice",
        "instructions": "Classify the market regime from these stats.",
        "criteria": {
            "trending": "directional persistent move",
            "mean_reverting": "range bound, low efficiency ratio",
            "neutral": "no clear structure",
            "high_vol": "elevated volatility dominant",
        },
    },
    "conviction": {
        "type": "score",
        "instructions": "Conviction for this setup.",
        "criteria": ["0-3 low", "4-6 medium", "7-10 high"],
    },
    "sanity": {
        "type": "noul",
        "instructions": "Is this a coherent trade setup?",
    },
}

# Post-trade failure-mode questions for forensics second opinions.
LAYA_FORENSICS_QUESTIONS = {
    "failure_mode": {
        "type": "choice",
        "instructions": "Classify why this trade won or lost.",
        "criteria": {
            "TIMING": "entered too soon or exited too early",
            "REGIME_MISMATCH": "traded against the prevailing regime",
            "SIZING": "position size or stop placement wrong for the move",
            "SIGNAL_QUALITY": "entry signal was weak or miscalibrated",
            "STOP_HUNT": "stopped out despite good signal, hunt or slippage",
            "STALE_POSITION": "held too long without hitting TP/SL",
            "GOOD_EXECUTION": "profitable, replicable setup",
            "UNKNOWN": "insufficient data to classify",
        },
    },
    "lesson_hint": {
        "type": "noul",
        "instructions": "Does this trade summary form a coherent failure story?",
    },
}

_FAILURE_MODES = set(LAYA_FORENSICS_QUESTIONS["failure_mode"]["criteria"])


def _regime_family(regime_str: str) -> set:
    """Map an engine regime label to the laya regime vocabulary families."""
    r = (regime_str or "").upper()
    fam = set()
    if "TRENDING" in r:
        fam.add("trending")
    if "MEAN_REVERT" in r:
        fam.add("mean_reverting")
    if "NEUTRAL" in r:
        fam.add("neutral")
    if "HIGH_VOL" in r:
        fam.add("high_vol")
    return fam or {"neutral"}


def interpret_entry_verdict(verdict: Dict, engine_regime: str = "",
                            veto_prob: float = LAYA_VETO_PROB) -> Dict:
    """
    Normalize a raw laya /predict payload into an entry-decision summary.

    Returns:
      entry, enter_prob, avoid_prob, regime, regime_prob, regime_mismatch,
      conviction_norm, noul, veto, multiplier, summary
    """
    answers = (verdict or {}).get("answers", {}) or {}

    entry_ans = answers.get("entry", {}) or {}
    entry = entry_ans.get("choice")
    probs = entry_ans.get("probabilities", {}) or {}
    enter_prob = float(probs.get("enter", 0.0))
    avoid_prob = float(probs.get("avoid", 0.0))

    regime_ans = answers.get("regime", {}) or {}
    regime = regime_ans.get("choice")
    regime_probs = regime_ans.get("probabilities", {}) or {}
    regime_prob = float(regime_probs.get(regime, 0.0)) if regime else 0.0
    engine_fam = _regime_family(engine_regime)
    regime_mismatch = bool(
        regime and engine_fam and regime_prob >= 0.55 and regime not in engine_fam
    )

    conv_ans = answers.get("conviction", {}) or {}
    # conviction.score is an expected value over legend indices 0..2
    conviction_norm = max(0.0, min(1.0, float(conv_ans.get("score", 0.0)) / 2.0))

    san_ans = answers.get("sanity", {}) or {}
    noul = float(san_ans.get("noul", san_ans.get("confidence", 0.5)) or 0.5)

    # Veto: strong avoid probability. Never creates entries (quant-only policy).
    veto = (entry == "avoid" and avoid_prob >= veto_prob)

    # Bounded multiplier around meta-confidence: conviction + coherence blend,
    # with a mild penalty when the System-1 model reads a different regime family.
    blend = 0.5 * conviction_norm + 0.5 * max(0.0, min(1.0, noul))
    multiplier = 0.92 + 0.16 * blend
    if regime_mismatch:
        multiplier *= 0.95
    multiplier = max(0.85, min(1.08, multiplier))

    summary = (f"entry={entry}(p={avoid_prob:.2f}/enter={enter_prob:.2f}) "
               f"regime={regime}(p={regime_prob:.2f}) "
               f"mismatch={regime_mismatch} conviction={conviction_norm:.2f} "
               f"noul={noul:.2f} mult={multiplier:.3f}")

    return {
        "entry": entry,
        "enter_prob": enter_prob,
        "avoid_prob": avoid_prob,
        "regime": regime,
        "regime_prob": regime_prob,
        "regime_mismatch": regime_mismatch,
        "conviction_norm": conviction_norm,
        "noul": noul,
        "veto": veto,
        "multiplier": multiplier,
        "summary": summary,
    }


def extract_failure_mode(verdict: Dict) -> Optional[Dict]:
    """Pull the failure_mode choice out of a forensics verdict."""
    if not verdict:
        return None
    ans = (verdict.get("answers", {}) or {}).get("failure_mode", {}) or {}
    choice = ans.get("choice")
    probs = ans.get("probabilities", {}) or {}
    if not choice or choice not in _FAILURE_MODES:
        return None
    return {"mode": choice, "prob": float(probs.get(choice, 0.0))}


class LayaClient:
    """Thread-safe, non-blocking client with cache, queue worker, circuit breaker."""

    def __init__(self, url: str = LAYA_URL, enabled: bool = LAYA_ENABLED,
                 cache_ttl: float = LAYA_CACHE_TTL, timeout: float = LAYA_TIMEOUT):
        self.url = url
        # /predict -> /feedback on the same origin
        if url.rstrip("/").endswith("/predict"):
            self.feedback_url = url.rstrip("/")[: -len("/predict")] + "/feedback"
        else:
            self.feedback_url = url.rstrip("/") + "/feedback"
        self.enabled = enabled
        self.cache_ttl = cache_ttl
        self.timeout = timeout
        self._cache: Dict[str, Dict] = {}          # key -> {"result": dict, "ts": float}
        self._pending: set = set()                 # keys currently queued
        self._queue: "queue.Queue" = queue.Queue(maxsize=16)
        self._lock = threading.Lock()
        self._worker = None
        self._fail_count = 0
        self._circuit_open_until = 0.0
        self._last_error = ""
        self.stats = {"served": 0, "enqueued": 0, "dropped": 0, "failures": 0,
                      "circuit_opens": 0, "sink_errors": 0,
                      "feedback_sent": 0, "feedback_errors": 0}

    # ---------- public API ----------

    def evaluate(self, state: Dict, questions: Dict, cache_key: str,
                 sink: Optional[Callable[[Dict, Dict], None]] = None,
                 store: bool = True) -> Optional[Dict]:
        """
        Non-blocking evaluation. Returns a FRESH cached verdict if available,
        otherwise enqueues a background prediction (deduped by cache_key) and
        returns None. Never raises. store=False skips the cache (fire-and-forget
        sink-only jobs, e.g. unique per-trade forensics opinions).
        """
        if not self.enabled:
            return None
        now = time.time()
        if now < self._circuit_open_until:
            return None
        if store:
            with self._lock:
                hit = self._cache.get(cache_key)
                if hit and (now - hit["ts"]) < self.cache_ttl:
                    self.stats["served"] += 1
                    return hit["result"]
        # Stale or missing -> enqueue for background processing.
        if cache_key in self._pending:
            return None
        self._ensure_worker()
        with self._lock:
            self._pending.add(cache_key)
        try:
            self._queue.put_nowait({"state": state, "questions": questions,
                                    "key": cache_key, "sink": sink, "store": store})
            self.stats["enqueued"] += 1
        except queue.Full:
            with self._lock:
                self._pending.discard(cache_key)
            self.stats["dropped"] += 1
        return None

    def evaluate_blocking(self, state: Dict, questions: Dict,
                          timeout: float = 8.0) -> Optional[Dict]:
        """Direct synchronous prediction with timeout. Never raises. For the rare
        paths where a few seconds are acceptable (e.g. UNKNOWN forensics)."""
        if not self.enabled or time.time() < self._circuit_open_until:
            return None
        try:
            result = self._post(state, questions, timeout)
            self._note_success()
            return result
        except Exception as e:
            self._note_failure(str(e))
            return None

    def submit_feedback(self, decision_id: str, question_id: str,
                        ground_truth: Any, reward: float,
                        target_type: str = "choice", notes: str = "") -> bool:
        """
        POST ground truth back to Laya's /feedback learning endpoint so the
        engine calibrates from our realized trade outcomes. Never raises.
        """
        if not self.enabled or not decision_id:
            return False
        if time.time() < self._circuit_open_until:
            return False
        payload = json.dumps({
            "decision_id": decision_id,
            "question_id": question_id,
            "ground_truth": ground_truth,
            "target_type": target_type,
            "reward": reward,
            "notes": notes[:500],
        }).encode("utf-8")
        try:
            req = urllib.request.Request(self.feedback_url, data=payload,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                ok = getattr(resp, "status", 200) == 200
                if ok:
                    self.stats["feedback_sent"] = self.stats.get("feedback_sent", 0) + 1
                    return True
                return False
        except Exception as e:
            self.stats["feedback_errors"] = self.stats.get("feedback_errors", 0) + 1
            log.debug(f"[LAYA] feedback failed: {e}")
            return False

    def health(self) -> Dict:
        with self._lock:
            fresh = sum(1 for v in self._cache.values()
                        if time.time() - v["ts"] < self.cache_ttl)
            return {
                "enabled": self.enabled,
                "url": self.url,
                "circuit_open": time.time() < self._circuit_open_until,
                "fresh_cached": fresh,
                "queued": self._queue.qsize(),
                "last_error": self._last_error,
                **self.stats,
            }

    # ---------- internals ----------

    def _ensure_worker(self):
        if self._worker and self._worker.is_alive():
            return
        with self._lock:
            if self._worker and self._worker.is_alive():
                return
            self._worker = threading.Thread(target=self._worker_loop,
                                            name="laya-worker", daemon=True)
            self._worker.start()

    def _worker_loop(self):
        while True:
            try:
                job = self._queue.get(timeout=30)
            except queue.Empty:
                continue
            key = job["key"]
            try:
                if time.time() < self._circuit_open_until:
                    continue
                result = self._post(job["state"], job["questions"], self.timeout)
                with self._lock:
                    if job.get("store", True):
                        self._cache[key] = {"result": result, "ts": time.time()}
                    self._pending.discard(key)
                self._note_success()
                sink = job.get("sink")
                if sink:
                    try:
                        sink(result, job["state"])
                    except Exception as se:
                        self.stats["sink_errors"] += 1
                        log.debug(f"[LAYA] sink error: {se}")
            except Exception as e:
                with self._lock:
                    self._pending.discard(key)
                self._note_failure(str(e))

    def _post(self, state: Dict, questions: Dict, timeout: float) -> Dict:
        payload = json.dumps({"state": state, "questions": questions}).encode("utf-8")
        req = urllib.request.Request(self.url, data=payload,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _note_success(self):
        with self._lock:
            self._fail_count = 0
            self._last_error = ""

    def _note_failure(self, err: str):
        with self._lock:
            self._fail_count += 1
            self.stats["failures"] += 1
            self._last_error = err[:200]
            if self._fail_count >= LAYA_CIRCUIT_AFTER:
                self._circuit_open_until = time.time() + LAYA_CIRCUIT_COOLDOWN
                self.stats["circuit_opens"] += 1
                self._fail_count = 0
                log.warning(f"[LAYA] circuit opened for {LAYA_CIRCUIT_COOLDOWN:.0f}s "
                            f"after repeated failures: {err[:120]}")


_client: Optional[LayaClient] = None
_client_lock = threading.Lock()


def get_laya_client() -> LayaClient:
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = LayaClient()
                if _client.enabled:
                    log.info(f"[LAYA] client ready -> {_client.url} "
                             f"(async cache, veto_p={LAYA_VETO_PROB})")
                else:
                    log.info("[LAYA] client disabled (LAYA_ENABLED=false)")
    return _client
