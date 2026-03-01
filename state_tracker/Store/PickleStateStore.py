from __future__ import annotations

import pickle
import threading
from pathlib import Path
from typing import Optional

from state_tracker.Models import StateSnapshot
from state_tracker.Store.StateStore import StateStore
from state_tracker.errors import StoreContractError


class PickleStateStore(StateStore):
    """
    Thread-safe pickle-based state store.
    Stores one latest StateSnapshot.
    """

    def __init__(self, filename: str = "state.pkl") -> None:
        self.filename = Path(filename)
        self._lock = threading.RLock()

    def save(self, snapshot: StateSnapshot) -> None:
        if not isinstance(snapshot, StateSnapshot):
            raise StoreContractError("Only StateSnapshot instances can be saved")

        with self._lock:
            try:
                self.filename.parent.mkdir(parents=True, exist_ok=True)
                tmp_file = self.filename.with_suffix(".tmp")
                with tmp_file.open("wb") as f:
                    pickle.dump(snapshot.to_dict(), f, protocol=pickle.HIGHEST_PROTOCOL)
                tmp_file.replace(self.filename)
            except Exception as exc:
                raise StoreContractError(f"Failed to save snapshot to {self.filename}") from exc

    def load(self) -> Optional[StateSnapshot]:
        with self._lock:
            if not self.filename.exists():
                return None
            try:
                with self.filename.open("rb") as f:
                    raw_data = pickle.load(f)
                return StateSnapshot.from_mapping(raw_data)
            except Exception as exc:
                raise StoreContractError(f"Failed to load snapshot from {self.filename}") from exc

    def reset(self) -> None:
        with self._lock:
            if self.filename.exists():
                try:
                    self.filename.unlink()
                except Exception as exc:
                    raise StoreContractError(f"Failed to reset store {self.filename}") from exc

    @property
    def stop_flag(self) -> bool:
        snapshot = self.load()
        if snapshot is None:
            return False
        return bool(snapshot.states.get("stop_flag", False))

    @stop_flag.setter
    def stop_flag(self, value: bool) -> None:
        snapshot = self.load()
        if snapshot is None:
            raise StoreContractError("Cannot set stop_flag without existing snapshot")

        new_states = dict(snapshot.states)
        new_states["stop_flag"] = bool(value)

        updated = StateSnapshot(
            run_id=snapshot.run_id,
            mode=snapshot.mode,
            states=new_states,
            captured_at_utc=snapshot.captured_at_utc,
            schema_version=snapshot.schema_version,
            metadata=dict(snapshot.metadata),
        )
        self.save(updated)
