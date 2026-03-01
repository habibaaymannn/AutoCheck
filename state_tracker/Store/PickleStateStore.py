from __future__ import annotations

import pickle
import threading
from pathlib import Path
from typing import Optional

from state_tracker.Models import StateSnapshot
from state_tracker.errors import StoreContractError
from Store.StateStore import StateStore

class PickleStateStore(StateStore):
    """
    Thread-safe pickle-based state store.
    Stores a full StateSnapshot object.
    """
    def __init__(self,filename:str="state,pkl")->None:
        self.filename=Path(filename)
        self._lock=threading.RLock()

    #Public API
    def save(self, snapshot:StateSnapshot)->None:
        #Persist a StateSnapshot atomically.
        if not isinstance(snapshot,StateSnapshot):
            raise StoreContractError("Only StateSnapshot instances can be saved")
        with self._lock:
            try:
                tmp_file=self.filename.with_suffix(".tmp")

                #write tmp then replace
                with tmp_file.open("wb")as f:
                    pickle.dump(snapshot.to_dict(),f)

                tmp_file.replace(self.filename)

            except Exception as exc:
                raise StoreContractError(
                    f"Failed to save snapshot to {self.filename}"
                ) from exc
    
    def load(self)->Optional[StateSnapshot]:
        #Load latest snapshot if exists.
        with self._lock:
            if not self.filename.exists():
                return None
            
            try:
                with self.filename.open("rb") as f:
                    raw_data=pickle.load(f)
                return StateSnapshot.from_mapping(raw_data)
            except Exception as exc:
                raise StoreContractError(
                    f"Failed to load snapshot from {self.filename}"
                ) from exc
            
    def reset(self)->None:
        #Delete stored state.
        with self._lock:
            if self.filename.exists():
                try:
                    self.filename.unlink()
                except Exception as exc:
                    raise StoreContractError(
                        f"Failed to reset store {self.filename}"
                    ) from exc
                

    # Optional HPC stop flag
    @property
    def stop_flag(self)->bool:
        snapshot = self.load()
        if snapshot and "stop_flag" in snapshot.states:
            return bool(snapshot.states["stop_flag"])
        return False
    
    @stop_flag.setter
    def stop_flag(self, value: bool) -> None:
        snapshot = self.load()
        if snapshot is None:
            raise StoreContractError("Cannot set stop_flag without existing snapshot")

        snapshot.states["stop_flag"] = bool(value)
        self.save(snapshot)