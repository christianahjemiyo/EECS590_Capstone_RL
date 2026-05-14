from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _flatten(prefix: str, obj: Any, out: dict[str, np.ndarray]) -> None:
    if isinstance(obj, np.ndarray):
        out[prefix] = obj
        return
    if isinstance(obj, (list, tuple)):
        out[prefix] = np.asarray(obj)
        return
    if isinstance(obj, (int, float, bool, np.integer, np.floating)):
        out[prefix] = np.asarray(obj)
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            child = f"{prefix}__{key}" if prefix else str(key)
            _flatten(child, value, out)
        return
    out[prefix] = np.asarray(str(obj))


def save_npz_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat: dict[str, np.ndarray] = {}
    _flatten("", payload, flat)
    np.savez(path, **flat)
