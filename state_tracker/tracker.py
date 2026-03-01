from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict

from state_tracker.store import StateStore

CONTROL_FIELDS: frozenset = frozenset(
    {"session_started_at", "session_elapsed_seconds", "max_session_time"})


class StateTracker(ABC):
    def __init__(self, store: StateStore, lock: Lock, validator) -> None:
        self._store = store
        self._lock = lock
        self._validator = validator
        self._stop_flag: bool = False

    @abstractmethod
    def update_state(self, **kwargs: Any) -> Dict[str, Any]:
        """
        Validate and persist state fields.
        different implementation for ML and HPC
        """

    @abstractmethod
    def get_snapshot_data(self) -> Dict[str, Any]:
        """
        Assemble the checkpoint / resume payload.
        different implementation for ML and HPC
        """

    def get_state(self) -> Dict[str, Any]:
        """Return a copy of all stored user state plus stop_flag."""
        with self._lock:
            return self._build_state_snapshot()

    def update_session_time(self) -> float:
        """Compute and persist elapsed session time in seconds.

        First call records session start and returns 0.0.
        Subsequent calls return seconds elapsed since that start.
        """
        with self._lock:
            now = datetime.now(timezone.utc)
            start = self._store.get("session_started_at")

            if start is None:
                self._store.set("session_started_at", now)
                self._store.set("session_elapsed_seconds", 0.0)
                return 0.0

            if isinstance(start, str):
                try:
                    start = datetime.fromisoformat(start)
                except ValueError:
                    pass

            if isinstance(start, datetime):
                if start.tzinfo is None:
                    start = start.replace(tzinfo=timezone.utc)
                elapsed = (now - start).total_seconds()
            else:
                try:
                    elapsed = now.timestamp() - float(start)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "session_started_at must be a datetime, ISO-8601 string, "
                        "or epoch seconds."
                    ) from exc

            elapsed = max(0.0, elapsed)
            self._validate_ml({"session_elapsed_seconds": elapsed})
            self._store.set("session_elapsed_seconds", elapsed)
            return elapsed

    def request_stop(self) -> None:
        """Set stop_flag to True."""
        with self._lock:
            self._stop_flag = True

    def should_training_stop(self) -> bool:
        """Return True if stop_flag is set or elapsed >= max_session_time."""
        with self._lock:
            if self._stop_flag:
                return True
            max_session_time = self._store.get("max_session_time")
            if max_session_time is None:
                return False
            elapsed = self._store.get("session_elapsed_seconds") or 0.0
            return float(elapsed) >= float(max_session_time)

    def _build_state_snapshot(self) -> Dict[str, Any]:
        state = self._store.all()
        state["stop_flag"] = self._stop_flag
        return state