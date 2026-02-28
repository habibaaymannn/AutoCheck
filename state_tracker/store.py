from __future__ import annotations

import copy
import threading
import time
from typing import Any, Dict

from .validation import validate_bulk_data, validate_key, validate_namespace, validate_snapshot


class StateTracker:
    """Thread-safe namespaced store used by runners and controllers."""

    def __init__(self) -> None:
        self._state: Dict[str, Dict[str, Any]] = {}
        self._version = 0
        self._lock = threading.RLock()
        self._last_update_time = time.time()

    # ----------------------------
    # Update Operations (Runner uses these)
    # ----------------------------
    def update(self, namespace: str, key: str, value: Any) -> None:
        validate_namespace(namespace)
        validate_key(key)
        with self._lock:
            if namespace not in self._state:
                self._state[namespace] = {}
            self._state[namespace][key] = value
            self._version += 1
            self._last_update_time = time.time()

    def bulk_update(self, namespace: str, data: Dict[str, Any]) -> None:
        validate_namespace(namespace)
        validate_bulk_data(data)
        with self._lock:
            if namespace not in self._state:
                self._state[namespace] = {}
            self._state[namespace].update(data)
            self._version += 1
            self._last_update_time = time.time()

    # ----------------------------
    # Read Operations (Controller uses these)
    # ----------------------------
    def get(self, namespace: str, key: str) -> Any:
        validate_namespace(namespace)
        validate_key(key)
        with self._lock:
            return self._state.get(namespace, {}).get(key)

    def get_namespace(self, namespace: str) -> Dict[str, Any]:
        validate_namespace(namespace)
        with self._lock:
            return copy.deepcopy(self._state.get(namespace, {}))

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "version": self._version,
                "timestamp": self._last_update_time,
                "state": copy.deepcopy(self._state),
            }

    # ----------------------------
    # Restore Operation (Controller triggers)
    # ----------------------------
    def restore(self, snapshot_data: Dict[str, Any]) -> None:
        validate_snapshot(snapshot_data)
        with self._lock:
            self._state = copy.deepcopy(snapshot_data["state"])
            self._version = int(snapshot_data["version"])
            self._last_update_time = time.time()
