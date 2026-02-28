from __future__ import annotations

from typing import List

from config.ConfigManager import ConfigManager
from config.YamlOBJ.HPC import HPC
from config.YamlOBJ.System import System
from config.YamlOBJ.enum import ExecutionMode
# from state_tracker.contracts import StateProvider, Store
from state_tracker.Models import TrackedStateSpec
from state_tracker.StateTracker import StateTracker


class HPCStateTracker(StateTracker):
    mode = ExecutionMode.HPC.value

    def __init__(
        self,
        *,
        run_id: str,
        # provider: StateProvider,
        tracked_states: List[TrackedStateSpec],
        # store: Store | None = None,
        scheduler: str | None = None,
    ) -> None:
        super().__init__(
            run_id=run_id,
            # provider=provider,
            tracked_states=tracked_states,
            # store=store,
        )
        self._scheduler = scheduler

    @property
    def tracker_kind(self) -> str:
        return "hpc"

    @property
    def scheduler(self) -> str | None:
        return self._scheduler

    @classmethod
    def from_config(
        cls,
        cm: ConfigManager,
        # *,
        # provider: StateProvider,
        # store: Store | None = None,
    ) -> "HPCStateTracker":
        if cm.mode != ExecutionMode.HPC.value:
            raise ValueError(f"Config mode is '{cm.mode}', expected '{ExecutionMode.HPC.value}'")

        system = cm.get(System)
        hpc_cfg = cm.get(HPC)

        specs = [
            TrackedStateSpec(name=s.name, type_name=s.type, source=s.source)
            for s in hpc_cfg.tracked_states
        ]

        return cls(
            run_id=system.run_id,
            # provider=provider,
            tracked_states=specs,
            # store=store,
            scheduler=system.fram_schd,
        )
