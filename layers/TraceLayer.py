from __future__ import annotations

import copy
import sys
import threading
from typing import Any, Dict, List, Set

from BaseLayer import BaseLayer
from logger import setup_logger


class TraceLayer(BaseLayer):
    """
    Combined trace + attr layer.

    1. TRACE (scalars) — uses sys.settrace to observe the user's script.
       On every function return inside target_file, reads f_locals and
       captures any variable whose name is in poll_target (e.g. epoch,
       batch_idx, global_step, loss). These are plain int/float values
       that can be stored directly.

    2. ATTR (objects) — on the same f_locals scan, looks for any object
       that has a .state_dict() method (model, optimizer, scheduler).
       Stores a live reference to each one. On snapshot(), calls
       .state_dict() to get current weights. On restore(), calls
       .load_state_dict() to push weights back into the live object.
    """

    def __init__(
            self,
            target_file: str,
            poll_target: List[str],
            snapshot_target: List[str],
            run_id: str = "default",
    ) -> None:
        """
        Args:
            target_file:     Absolute path to the user's script (train.py).
                             Only frames from this file are observed.
            poll_target:     Scalar field names to capture (e.g. ["epoch", "batch_idx"]).
                             Populated by RunnerScript._build_provider() from tracker fields.
            snapshot_target: Object field names to discover (e.g. ["model", "optimizer"]).
                             Used to filter which objects to hold references to.
            run_id:          For logging only.
        """
        super().__init__(poll_target=poll_target, snapshot_target=snapshot_target)
        self.logger = setup_logger(self.__class__.__name__, run_id)

        self.target_file: str = target_file
        self._watched_vars: Set[str] = set(poll_target)  # scalar names
        self._watched_objs: Set[str] = set(snapshot_target)  # object names

        self._captured: Dict[str, Any] = {}  # scalars: {"epoch": 7, "loss": 0.4}
        self._objects: Dict[str, Any] = {}  # live refs: {"model": <ResNet>, ...}

    def attach(self) -> None:
        """
        Install the global trace hook into the Python interpreter.
        Called once by Provider before the user's script runs via runpy.
        After this, every function call in the process goes through
        _global_trace — but only train.py frames get a local tracer.
        """
        with self._lock:
            sys.settrace(self._global_trace)
            threading.settrace(self._global_trace)  # covers new threads too
            self._set_active(True)
            self.logger.info(
                f"[ATTACH] | TraceLayer active | "
                f"target={self.target_file} | "
                f"watching scalars={self._watched_vars} | "
                f"watching objects={self._watched_objs}"
            )

    def detach(self) -> None:
        """
        Remove the trace hook and release all references.
        Called when training ends or AutoCheck shuts down.
        """
        with self._lock:
            sys.settrace(None)
            threading.settrace(None)
            self._captured.clear()
            self._objects.clear()
            self._set_active(False)
            self.logger.info("[DETACH] | TraceLayer detached | all references released")

    def poll(self) -> Dict[str, Any]:
        """
        Lightweight read — returns current scalar values only.
        Does NOT call .state_dict() — safe to call frequently.
        Used by controller to check epoch/step progress between checkpoints.
        """
        with self._lock:
            return dict(self._captured)

    def snapshot(self) -> Dict[str, Any]:
        """
        Full checkpoint read — scalars + serialized object states.
        Calls .state_dict() on every discovered object.
        Returns a deep copy so training can continue safely while saving.
        Called only when a checkpoint is actually triggered.
        """
        with self._lock:
            result: Dict[str, Any] = dict(self._captured)  # scalars first

            for name, obj in self._objects.items():
                try:
                    result[name] = copy.deepcopy(obj.state_dict())
                    self.logger.debug(f"[SNAPSHOT] | serialized {name}")
                except Exception as e:
                    self.logger.error(f"[SNAPSHOT] | failed to serialize {name} | reason={e}")
                    raise

            self.logger.info(
                f"[SNAPSHOT] | complete | "
                f"scalars={list(self._captured.keys())} | "
                f"objects={list(self._objects.keys())}"
            )
            return result

    def restore(self, snapshot: Dict[str, Any]) -> None:
        """
        Push saved object states back into live objects via .load_state_dict().
        Scalars (epoch, batch_idx) are NOT restored here — the controller
        handles those by telling the loop where to resume from.
        Called on job resume before training restarts.
        """
        with self._lock:
            for name, obj in self._objects.items():
                if name in snapshot:
                    try:
                        obj.load_state_dict(snapshot[name])
                        self.logger.info(f"[RESTORE] | restored {name}")
                    except Exception as e:
                        self.logger.error(f"[RESTORE] | failed to restore {name} | reason={e}")
                        raise
                else:
                    self.logger.warning(f"[RESTORE] | {name} not found in snapshot — skipped")

    # ------------------------------------------------------------------
    # Trace hook internals
    # ------------------------------------------------------------------

    def _global_trace(self, frame: Any, event: str, arg: Any) -> Any:
        """
        Installed as the global trace function via sys.settrace.
        Fires on every function CALL in the entire process.

        Filter: only install a local tracer for frames coming from
        the user's target_file. Everything else (PyTorch internals,
        numpy, AutoCheck itself) returns None immediately — free.
        """
        if frame.f_code.co_filename == self.target_file:
            return self._local_trace
        return None

    def _local_trace(self, frame: Any, event: str, arg: Any) -> Any:
        """
        Local trace function installed only on target_file frames.
        Fires on call, line, return, exception events within those frames.

        We only act on RETURN — at that point f_locals is fully
        populated with the final values of all local variables.
        Ignoring line events means no per-line overhead.
        """
        if event == "return":
            self._extract(frame)
        return self._local_trace

    def _extract(self, frame: Any) -> None:
        """
        Read f_locals from a returning frame and update both stores.

          - Scalars: if name is in _watched_vars and value is int/float
                     → store in _captured directly
          - Objects: if value has .state_dict() and name is in _watched_objs
                     → store live reference in _objects (once only)

        Only updates _captured if at least one critical counter
        (epoch or global_step) is present — avoids storing partial
        state from helper functions.
        """
        locals_ = frame.f_locals
        found_scalars: Dict[str, Any] = {}
        found_objects: Dict[str, Any] = {}

        for name, value in locals_.items():
            # scalar capture
            if name in self._watched_vars and isinstance(value, (int, float)):
                found_scalars[name] = value

            # object discovery — only once per object name
            if (
                    name in self._watched_objs
                    and name not in self._objects
                    and hasattr(value, "state_dict")
                    and callable(getattr(value, "state_dict"))
            ):
                found_objects[name] = value

        # only update scalars if we found something meaningful
        # (epoch or global_step present = we're in the training loop)
        if "epoch" in found_scalars or "global_step" in found_scalars:
            with self._lock:
                self._captured.update(found_scalars)
                if found_scalars:
                    self.logger.debug(f"[EXTRACT] | scalars updated | {found_scalars}")

        # register newly discovered objects
        if found_objects:
            with self._lock:
                for name, obj in found_objects.items():
                    self._objects[name] = obj
                    self.logger.info(
                        f"[EXTRACT] | object discovered | "
                        f"name={name} | type={type(obj).__name__}"
                    )

    def is_ready(self) -> bool:
        """
        Returns True once at least one object has been discovered.
        Provider uses this to know it's safe to take a full snapshot.
        Before this returns True, snapshot() would return empty object states.
        """
        with self._lock:
            return len(self._objects) > 0

    def discovered_objects(self) -> List[str]:
        """Return names of all currently discovered objects. For logging/debug."""
        with self._lock:
            return list(self._objects.keys())
