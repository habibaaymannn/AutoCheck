from datetime import datetime, timezone
from threading import Lock
from typing import Any, Dict, Optional
from state_tracker.provider_interface import StateProvider as ProviderInterface
from state_tracker.store import StateStore as Store

class StateTracker:

    _SUPPORTED_MODES = {"ml", "hpc"} # todo will get it from config.yaml
    _CONTROL_FIELDS = {"session_started_at", "session_elapsed_seconds", "max_session_time"} # todo implement for ml and hpc seperately

    def __init__(
        self,
        store: Optional[Store] = None,
        # validator: Optional[Validator] = None, # when the validator is implemented
        provider: Optional[ProviderInterface] = None,
        mode: str = "ml",
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.store = store if store is not None else Store()
        # self.validator = validator if validator is not None else Validator()
        self.provider = provider
        self.lock = Lock()
        self.mode = str(mode).lower()
        if self.mode not in self._SUPPORTED_MODES:
            raise ValueError(
                f"Unsupported mode '{mode}'. Supported modes: {sorted(self._SUPPORTED_MODES)}"
            )

        if not hasattr(self.store, "stop_flag"):
            setattr(self.store, "stop_flag", False)

        if initial_state:
            self.update_state(**initial_state)

    def update_state(self, **kwargs: Any) -> Dict[str, Any]:
        with self.lock:
            updates = dict(kwargs)
            stop_flag_update = updates.pop("stop_flag", None)

            if stop_flag_update is not None and not isinstance(stop_flag_update, bool):
                raise TypeError(f"stop_flag must be bool, got {type(stop_flag_update)}")

            if self.mode == "hpc":
                control_data = {k: v for k, v in updates.items() if k in self._CONTROL_FIELDS}
                hpc_data = {k: v for k, v in updates.items() if k not in self._CONTROL_FIELDS}
                self._validate_ml(control_data)
                self._validate_hpc(hpc_data)
            else:
                self._validate_ml(updates)

            for key, value in updates.items():
                self._store_set(key, value)
            if stop_flag_update is not None:
                setattr(self.store, "stop_flag", stop_flag_update)
            return self._build_state_response()

    def get_state(self) -> Dict[str, Any]:
        with self.lock:
            return self._build_state_response()

    def get_snapshot_data(self) -> Dict[str, Any]:
        with self.lock:
            user_state = self._store_get_all()
            model_state = self._call_provider_method("get_model_state")
            if model_state is None:
                model_state = self._call_provider_method("collect_state")

            resume_state = {
                "model_state": model_state,
                "optimizer_state": self._call_provider_method("get_optimizer_state"),
                "rng_state": self._call_provider_method("get_rng_state"),
                "scheduler_state": self._call_provider_method("get_scheduler_state"),
            }
            return {**user_state, "stop_flag": self._get_stop_flag(), **resume_state}

    def update_session_time(self) -> float:
        with self.lock:
            now = datetime.now(timezone.utc)
            start = self._store_get("session_started_at")

            if start is None:
                self._store_set("session_started_at", now)
                self._store_set("session_elapsed_seconds", 0.0)
                return 0.0

            if isinstance(start, str):
                try:
                    start = datetime.fromisoformat(start)
                except ValueError:
                    pass

            if isinstance(start, datetime):
                elapsed = (now - start).total_seconds()
            else:
                try:
                    elapsed = float(now.timestamp()) - float(start)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "session_started_at must be datetime, ISO-8601 datetime string, or epoch seconds"
                    ) from exc

            elapsed = max(0.0, elapsed)
            self._validate_ml({"session_elapsed_seconds": elapsed})
            self._store_set("session_elapsed_seconds", elapsed)
            return elapsed

    def request_stop(self) -> None:
        with self.lock:
            setattr(self.store, "stop_flag", True)

    def should_training_stop(self) -> bool:
        with self.lock:
            if self._get_stop_flag():
                return True

            max_session_time = self._store_get("max_session_time")
            elapsed = self._store_get("session_elapsed_seconds") or 0.0
            if max_session_time is None:
                return False
            return float(elapsed) >= float(max_session_time)

    def _build_state_response(self) -> Dict[str, Any]:
        state = self._store_get_all()
        state["stop_flag"] = self._get_stop_flag()
        return state

    def _store_set(self, key: str, value: Any) -> None:
        setter = getattr(self.store, "set", None)
        if not callable(setter):
            raise AttributeError("Store implementation must provide a callable 'set(key, value)'")
        setter(key, value)

    def _store_get(self, key: str) -> Any:
        getter = getattr(self.store, "get", None)
        if callable(getter):
            try:
                return getter(key)
            except TypeError:
                return getter(key, None)
        return self._store_get_all().get(key)

    def _store_get_all(self) -> Dict[str, Any]:
        getter = getattr(self.store, "get_all", None)
        if callable(getter):
            return dict(getter())

        legacy_getter = getattr(self.store, "all", None)
        if callable(legacy_getter):
            return dict(legacy_getter())

        current_state = getattr(self.store, "current_state", None)
        if isinstance(current_state, dict):
            return dict(current_state)

        raise AttributeError(
            "Store implementation must provide one of: get_all(), all(), or current_state dict"
        )

    def _get_stop_flag(self) -> bool:
        if hasattr(self.store, "stop_flag"):
            return bool(getattr(self.store, "stop_flag"))
        return bool(self._store_get("stop_flag"))

    def _validate_ml(self, data: Dict[str, Any]) -> None:
        if not data:
            return
        validator_fn = getattr(self.validator, "validate_ml", None)
        if callable(validator_fn):
            validator_fn(data)

    def _validate_hpc(self, data: Dict[str, Any]) -> None:
        if not data:
            return
        validator_fn = getattr(self.validator, "validate_hpc", None)
        if callable(validator_fn):
            validator_fn(data)

    def _call_provider_method(self, method_name: str) -> Any:
        if self.provider is None:
            return None
        method = getattr(self.provider, method_name, None)
        if callable(method):
            return method()
        return None
