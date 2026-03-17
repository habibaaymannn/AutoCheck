from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseLayer(ABC):
    """
    Abstract base class for all state-capture layers.

    Every layer has two target lists that tell it what to look for:
      - poll_target:     fields needed frequently (scalars like epoch, batch_idx)
      - snapshot_target: fields needed at checkpoint time (objects like model, optimizer)

    These are set by RunnerScript._build_provider() at startup — the layer
    never reads the tracker directly.
    """

    def __init__(
        self,
        poll_target: List[str],
        snapshot_target: List[str],
    ) -> None:
        self._active: bool = False
        self.poll_target: List[str] = poll_target
        self.snapshot_target: List[str] = snapshot_target
        self._lock: threading.RLock = threading.RLock()
        self._pending_restore: Dict[str, Any] = {}

    @abstractmethod
    def attach(self) -> None:
        """
        Start observing the user's code.
        Called once by Provider before the user's script runs.
        """

    @abstractmethod
    def detach(self) -> None:
        """
        Stop observing. Release any hooks or references.
        Called when training ends or AutoCheck shuts down.
        """

    @abstractmethod
    def poll(self) -> Dict[str, Any]:
        """
        Lightweight read — returns current scalar values only.
        Called frequently by the controller to check progress.
        Must NOT call .state_dict() — that's snapshot()'s job.
        """

    @abstractmethod
    def snapshot(self) -> Dict[str, Any]:
        """
        Full read — returns scalars AND serialized object states.
        Called only when a checkpoint is actually triggered.
        Must return a deep copy — safe to save while training continues.
        """

    @abstractmethod
    def restore(self, snapshot: Dict[str, Any]) -> None:
        """
        Push a saved snapshot back into the live objects.
        Called on resume. Scalars are handled by the controller —
        this method only restores object states via .load_state_dict().
        """

    def is_active(self) -> bool:
        """Return True if the layer is currently attached and observing."""
        return self._active

    def _set_active(self, value: bool) -> None:
        with self._lock:
            self._active = value