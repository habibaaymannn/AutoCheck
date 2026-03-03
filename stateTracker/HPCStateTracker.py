from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Mapping, Optional, Sequence

from config.YamlOBJ.HPCState import HPCState
from enums import CheckpointMethod
from error import (
    InvalidTrackedStateSpecError,

)
from stateTracker.BaseTracker import BaseTracker

class HPCStateTracker(BaseTracker):
    def __init__(
            self,
            run_id:str,
            method:str,
            tracked_states:Sequence[HPCState],
            scheduler: Optional[str] = None,
            ):
        super().__init__(run_id=run_id, method=method)
        self.scheduler: Optional[str] = scheduler
        self.states: list[HPCState] = list(tracked_states)
        self.tracked_states: Dict[str, Any] = {}

        self.iteration: int=0
        self.last_completed_unit: int=0
        self.scheduler_status: str='unknown'
        self.latest_checkpoint_path: Optional[str]=None

        self.validate()

    def set_states(self, tracked_states: Sequence[HPCState]) -> None:
        with self.lock:
            self.states = list(tracked_states)
            self.validate()
    
    # validate there is states to be tracked, all states are in HPC state format & no duplicate
    def validate(self) -> bool:
        if not self.states:
            raise InvalidTrackedStateSpecError("No tracked states configured")

        allowed_methods = {m.value for m in CheckpointMethod}
        if self.method not in allowed_methods:
            raise InvalidTrackedStateSpecError(
                f"Unsupported checkpoint method '{self.method}'. Allowed: {sorted(allowed_methods)}"
            )

        names = set()
        for s in self.states:
            if not isinstance(s, HPCState):
                raise InvalidTrackedStateSpecError("All specs must be HPC State")
            if s.name in names:
                raise InvalidTrackedStateSpecError(f"Duplicate state name: {s.name}")
            names.add(s.name)
        return True
