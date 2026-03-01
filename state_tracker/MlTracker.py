from __future__ import annotations

from threading import Lock
from typing import Any, Dict

from state_tracker.tracker import StateTracker
from state_tracker.store import StateStore


class MLTracker(StateTracker):
    """Tracker for ML workloads (PyTorch, TensorFlow, …)."""

    def __init__(self, store: StateStore, lock: Lock, validator, provider=None) -> None:
        super().__init__(store=store, lock=lock, validator=validator)
        self._provider = provider

    def update_state(self, **kwargs: Any) -> Dict[str, Any]:
        with self._lock:
            updates = dict(kwargs)

            stop_flag_update = updates.pop("stop_flag", None)
            if stop_flag_update is not None and not isinstance(stop_flag_update, bool):
                raise TypeError(
                    f"stop_flag must be bool, got {type(stop_flag_update).__name__}"
                )

            # self._validate_ml(updates)

            for key, value in updates.items():
                self._store.set(key, value)

            if stop_flag_update is not None:
                self._stop_flag = stop_flag_update

            return self._build_state_snapshot()

    def get_snapshot_data(self) -> Dict[str, Any]:
        with self._lock:
            user_state = self._store.all()

            model_state = self._call_provider("get_model_state")
            if model_state is None:
                model_state = self._call_provider("collect_state")

            return {
                **user_state,
                "stop_flag": self._stop_flag,
                "model_state": model_state,
                "optimizer_state": self._call_provider("get_optimizer_state"),
                "rng_state": self._call_provider("get_rng_state"),
                "scheduler_state": self._call_provider("get_scheduler_state"),
            }

    def _call_provider(self, method_name: str) -> Any:
        if self._provider is None:
            return None
        method = getattr(self._provider, method_name, None)
        return method() if callable(method) else None