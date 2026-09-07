import json
import os
from src.utils.logger import log


def load_json(path, default=None):
    """Restore a persisted JSON value (e.g. a rolling-window history list).
    Missing/corrupt file -> default (a fresh mutable default is returned, never shared)."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default if default is not None else []


def save_json(path, value):
    """Atomically persist a JSON-serializable value: write to a tmp file then rename, so a
    crash mid-write can never leave a truncated/corrupt state file behind."""
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(value, f)
        os.replace(tmp, path)
    except Exception as e:
        log.error(f"[persisted_state] save failed for {path}: {e}")


def load_float(path, default=0.0):
    """Restore a persisted float (e.g. cross-restart peak-equity high-water mark).
    Missing/corrupt file -> default."""
    try:
        with open(path) as f:
            return float(f.read().strip())
    except Exception:
        return default


def save_float(path, value, fmt="{:.4f}"):
    """Atomically persist a float: write to a tmp file then rename, so a crash mid-write
    can never leave a truncated/corrupt state file behind."""
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "w") as f:
            f.write(fmt.format(value))
        os.replace(tmp, path)
    except Exception as e:
        log.error(f"[persisted_state] save failed for {path}: {e}")
