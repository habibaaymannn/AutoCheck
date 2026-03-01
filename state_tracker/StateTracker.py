from __future__ import annotations

from abc import ABC, abstractmethod
from threading import RLock
from typing import Any, Dict, Mapping, Optional, Sequence

from config.YamlOBJ.enum import StateType
from state_tracker.Models import StateSnapshot, TrackedStateSpec
from state_tracker.Store.StateStore import StateStore
from state_tracker.errors import (
    InvalidTrackedStateSpecError,
    MissingStateValueError,
    StateProviderContractError,
    StateTypeCoercionError,
    StoreContractError,
)


class StateProvider(ABC):
    @abstractmethod
    def fetch(self, specs: Sequence[TrackedStateSpec]) -> Mapping[str, Any]:
        pass


class StateTracker(ABC):
    """
    Base tracker with decoupled provider/store.
    Provider can be attached later (Runner Script phase).
    """

    def __init__(
        self,
        *,
        run_id: str,
        tracked_states: Sequence[TrackedStateSpec],
        provider: Optional[StateProvider] = None,
        store: Optional[StateStore] = None,
    ) -> None:
        if not run_id or not isinstance(run_id, str):
            raise ValueError("run_id must be a non-empty string")

        self._run_id = run_id
        self._provider = provider
        self._store = store
        self._tracked_states = list(tracked_states)

        self._validate_tracked_specs(self._tracked_states)

        self._state_values: Dict[str, Any] = {}
        self._lock = RLock()

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def tracked_states(self) -> Sequence[TrackedStateSpec]:
        return tuple(self._tracked_states)

    @property
    @abstractmethod
    def mode(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def tracker_kind(self) -> str:
        raise NotImplementedError

    def set_provider(self, provider: StateProvider) -> None:
        if provider is None:
            raise ValueError("provider cannot be None")
        self._provider = provider

    def update_once(self, *, persist: bool = False) -> StateSnapshot:
        if self._provider is None:
            raise StateProviderContractError("No provider is attached to this tracker")

        raw_values = self._provider.fetch(self._tracked_states)
        return self.update_from_raw(raw_values, persist=persist)

    def update_from_raw(
        self,
        raw_values: Mapping[str, Any],
        *,
        persist: bool = False,
    ) -> StateSnapshot:
        updates = self._prepare_updates(raw_values)

        with self._lock:
            self._state_values.update(updates)
            snapshot = self.snapshot()

        if persist and self._store is not None:
            self.persist_snapshot(snapshot)

        return snapshot

    def snapshot(self) -> StateSnapshot:
        with self._lock:
            return StateSnapshot(
                run_id=self._run_id,
                mode=self.mode,
                states=dict(self._state_values),
            )

    def persist_snapshot(self, snapshot: Optional[StateSnapshot] = None) -> None:
        if self._store is None:
            raise StoreContractError("No store configured for this tracker")

        snap = snapshot if snapshot is not None else self.snapshot()
        try:
            self._store.save(snap)
        except Exception as exc:
            raise StoreContractError(
                f"Failed to persist snapshot for run_id={self._run_id}"
            ) from exc

    def get_state(self, name: str, default: Any = None) -> Any:
        with self._lock:
            return self._state_values.get(name, default)

    @property
    def current_state(self) -> Mapping[str, Any]:
        with self._lock:
            return dict(self._state_values)

    def _prepare_updates(self, raw_values: Mapping[str, Any]) -> Dict[str, Any]:
        if not isinstance(raw_values, Mapping):
            raise StateProviderContractError(
                "Provider/raw input must return mapping {state_name: raw_value}"
            )

        updates: Dict[str, Any] = {}
        for spec in self._tracked_states:
            if spec.name not in raw_values:
                raise MissingStateValueError(
                    f"Provider/raw input did not return required state '{spec.name}'"
                )
            updates[spec.name] = self._coerce_value(spec, raw_values[spec.name])

        return updates

    @staticmethod
    def _validate_tracked_specs(specs: Sequence[TrackedStateSpec]) -> None:
        if not specs:
            raise InvalidTrackedStateSpecError("tracked_states cannot be empty")

        names = set()
        allowed = {e.value for e in StateType}

        for spec in specs:
            if not spec.name.strip():
                raise InvalidTrackedStateSpecError("state name cannot be empty")
            if not spec.source.strip():
                raise InvalidTrackedStateSpecError(
                    f"state '{spec.name}' has empty source"
                )
            if spec.name in names:
                raise InvalidTrackedStateSpecError(
                    f"duplicate state name '{spec.name}'"
                )
            if spec.normalized_type() not in allowed:
                raise InvalidTrackedStateSpecError(
                    f"Invalid type '{spec.normalized_type()}' for state '{spec.name}'. "
                    f"Allowed: {allowed}"
                )
            names.add(spec.name)

    @staticmethod
    def _coerce_value(spec: TrackedStateSpec, raw_value: Any) -> Any:
        t = spec.normalized_type()

        if t == StateType.INT.value:
            try:
                return int(raw_value)
            except Exception as exc:
                raise StateTypeCoercionError(
                    f"State '{spec.name}' expected int, got {raw_value!r}"
                ) from exc

        if t == StateType.FLOAT.value:
            try:
                return float(raw_value)
            except Exception as exc:
                raise StateTypeCoercionError(
                    f"State '{spec.name}' expected float, got {raw_value!r}"
                ) from exc

        if t == StateType.STR.value:
            return str(raw_value)

        if t == StateType.BOOL.value:
            if isinstance(raw_value, bool):
                return raw_value
            if isinstance(raw_value, str):
                v = raw_value.strip().lower()
                if v in {"1", "true", "yes", "on"}:
                    return True
                if v in {"0", "false", "no", "off"}:
                    return False
            if isinstance(raw_value, (int, float)):
                return bool(raw_value)
            raise StateTypeCoercionError(
                f"State '{spec.name}' expected bool, got {raw_value!r}"
            )

        raise StateTypeCoercionError(
            f"Unsupported type '{spec.normalized_type()}' for state '{spec.name}'. "
            "Allowed: int, float, str, bool"
        )
