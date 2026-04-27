from __future__ import annotations
from typing import Any, Dict, Optional, List

from config.YamlOBJ.HPCState import HPCState
from Utilites.enums import CheckpointMethod
from Utilites.error import (
    InvalidTrackedStateSpecError,

)
from provider.Provider import Provider
from stateTracker.BaseTracker import BaseTracker
from Utilites.logger import setup_logger

class HPCStateTracker(BaseTracker):
    def __init__(
            self, method: str, program_path: str, tracked_states: List[HPCState], scheduler: Optional[str] = None, run_id: str = "default"):
        super().__init__(method=method, program_path=program_path, run_id=run_id)
        self.logger = setup_logger(self.__class__.__name__, run_id)
        self.scheduler: Optional[str] = scheduler
        self.states: list[HPCState] = list(tracked_states)
        self.tracked_states: Dict[str, Any] = {}

        self.iteration: int=0
        self.last_completed_unit: int=0
        self.scheduler_status: str='unknown'
        self.latest_checkpoint_path: Optional[str]=None
        self.validate()
        self.logger.info(f"HPCStateTracker initialized | run_id={run_id} | method={method}")

    def init_provider(self):
        poll_list = [str(self.method), "iteration", "last_completed_unit"]
        target_list = ["scheduler"]
        temp = [x.name for x in self.states]
        target_list.extend(temp)
        self.provider = Provider(self.program_path, self.method, poll_list, target_list)

    def set_states(self, tracked_states: List[HPCState]) -> None:
        with self.lock:
            self.states = list(tracked_states)
            self.validate()
            self.logger.info(f"Tracked states updated | states={[s.name for s in self.states]}")

    def update_ckpt_method(self) -> None:
        with self.lock:
            if self.provider is None:
                self.logger.error("Provider is not set in update_chpnt_method")
                raise RuntimeError("Provider is not set.")
            try:
                prov_state: Dict[str, Any] = self.provider.fetch_ckpt()
            except Exception:
                self.logger.exception("Failed to fetch provider state in update_chpnt_method")
                raise

            # Only update checkpoint-related progress fields
            self.iteration = prov_state.get("iteration", self.iteration)
            self.last_completed_unit = prov_state.get(
                "last_completed_unit", self.last_completed_unit
            )
            self.logger.info(
                f"Checkpoint fields updated | iteration={self.iteration} "
                f"last_completed_unit={self.last_completed_unit}"
            )

    def update_all_from_prov(self) -> None:
        with self.lock:
            if self.provider is None:
                self.logger.error("Provider is not set in update_all_from_prov")
                raise RuntimeError("Provider is not set.")
            try:
                prov_state: Dict[str, Any] = self.provider.fetch_all()
            except Exception:
                self.logger.exception("Failed to fetch provider state in update_all_from_prov")
                raise

            # update dynamic states
            for state in self.states:
                self.tracked_states[state.name] = prov_state.get(
                    state.name, self.tracked_states.get(state.name)
                )

            self.iteration = prov_state.get("iteration", self.iteration)
            self.last_completed_unit = prov_state.get(
                "last_completed_unit", self.last_completed_unit
            )
            self.scheduler_status = prov_state.get(
                "scheduler_status", self.scheduler_status
            )
            self.latest_checkpoint_path = prov_state.get(
                "latest_checkpoint_path", self.latest_checkpoint_path
            )
            self.logger.info(
                f"All HPC states updated from provider | iteration={self.iteration} "
                f"last_completed_unit={self.last_completed_unit} | scheduler_status={self.scheduler_status}"
            )

    def snapshot(self) -> Dict[str, Any]:
        self.update_all_from_prov()
        with self.lock:
            snap = {
                "run_id": self.run_id,
                "method": self.method,
                "scheduler": self.scheduler,
                "start_time": self.start_time.isoformat(),
                "iteration": self.iteration,
                "last_completed_unit": self.last_completed_unit,
                "scheduler_status": self.scheduler_status,
                "latest_checkpoint_path": self.latest_checkpoint_path,
                "tracked_states": dict(self.tracked_states),
            }
            self.logger.info(
                "Snapshot taken | iteration=%s | last_completed_unit=%s | tracked_count=%s",
                self.iteration, self.last_completed_unit, len(self.tracked_states)
            )
            self.logger.debug("Snapshot keys: %s", list(snap.keys()))
            return snap

    def set_all_from_ckpt(self, state: Dict[str, Any]) -> None:
        with self.lock:
            if not isinstance(state, dict):
                self.logger.error("Invalid checkpoint payload type: %s", type(state).__name__)
                raise RuntimeError("Checkpoint payload must be a dict")
            self.provider.restore(state)
            self.iteration = state.get("iteration", 0)
            self.last_completed_unit = state.get("last_completed_unit", 0)
            self.scheduler_status = state.get("scheduler_status", "unknown")
            self.latest_checkpoint_path = state.get("latest_checkpoint_path")

            saved_tracked = state.get("tracked_states", {})
            if isinstance(saved_tracked, dict):
                self.tracked_states = saved_tracked
            else:
                self.logger.warning("Invalid tracked_states in checkpoint (not dict). Resetting to empty.")
                self.tracked_states = {}
            self.logger.info(
                f"Tracker state restored from checkpoint | iteration={self.iteration} "
                f"last_completed_unit={self.last_completed_unit} | scheduler_status={self.scheduler_status}"
            )
            self.logger.debug("Checkpoint payload keys: %s", list(state.keys()))

    # validate there is states to be tracked, all states are in HPC state format & no duplicate
    def validate(self) -> bool:
        if not self.states:
            self.logger.error("Validation failed: no tracked states configured")
            raise InvalidTrackedStateSpecError("No tracked states configured")

        allowed_methods = {m.value for m in CheckpointMethod}
        if self.method not in allowed_methods:
            self.logger.error("Validation failed: unsupported checkpoint method '%s'", self.method)
            raise InvalidTrackedStateSpecError(
                f"Unsupported checkpoint method '{self.method}'. Allowed: {sorted(allowed_methods)}"
            )

        names = set()
        for s in self.states:
            if not isinstance(s, HPCState):
                self.logger.error("Validation failed: invalid state spec type %s", type(s).__name__)
                raise InvalidTrackedStateSpecError("All specs must be HPC State")
            if s.name in names:
                self.logger.error("Validation failed: duplicate state name '%s'", s.name)
                raise InvalidTrackedStateSpecError(f"Duplicate state name: {s.name}")
            names.add(s.name)
        self.logger.info(f"Validation passed | tracked_states={[s.name for s in self.states]} | method={self.method}")
        return True
