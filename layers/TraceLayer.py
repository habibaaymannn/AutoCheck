from __future__ import annotations

import copy
import ctypes
import sys
import threading
from typing import Any, Dict, List, Set

from layers.BaseLayer import BaseLayer
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
        self._pending_restore: Dict[str, Any] = {}

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
            self._pending_restore.clear()
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
            result: Dict[str, Any] = dict(self._captured)
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
        Arm _pending_restore with saved state for both scalars and objects.

        Scalars → written back into frames via PyFrame_LocalsToFast
                  the moment they appear in f_locals in _extract()
        Objects → .load_state_dict() called the moment the object
                  is discovered in f_locals by _extract()

        Nothing is applied immediately here — everything is deferred
        because the user's script hasn't run yet so _objects is empty
        at the time restore() is called.
        """
        with self._lock:
            for name in self._watched_vars:
                if name in snapshot:
                    self._pending_restore[name] = snapshot[name]
                    self.logger.info(f"[RESTORE] | armed scalar | {name}")
                else:
                    self.logger.warning(f"[RESTORE] | scalar {name} not found in snapshot — skipped")

            for name in self._watched_objs:
                if name in snapshot:
                    self._pending_restore[name] = snapshot[name]
                    self.logger.info(f"[RESTORE] | armed object | {name}")
                else:
                    self.logger.warning(f"[RESTORE] | object {name} not found in snapshot — skipped")

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
        if event in ("line", "return"):
            self._extract(frame)
        return self._local_trace

    def _extract(self, frame: Any) -> None:
        locals_ = frame.f_locals
        found_scalars: Dict[str, Any] = {}
        found_objects: Dict[str, Any] = {}

        for name, value in locals_.items():
            if name in self._watched_vars and isinstance(value, (int, float)):
                found_scalars[name] = value
            if (
                    name in self._watched_objs
                    and name not in self._objects
                    and hasattr(value, "state_dict")
                    and callable(getattr(value, "state_dict"))
            ):
                found_objects[name] = value

        if not found_scalars and not found_objects:
            return

        # ── RESTORE scalars ───────────────────────
        needs_sync = False
        with self._lock:
            for name in list(found_scalars.keys()):
                if name in self._pending_restore:
                    saved = self._pending_restore.pop(name)
                    frame.f_locals[name] = saved
                    found_scalars[name] = saved
                    needs_sync = True
                    self.logger.info(f"[RESTORE] | frame writeback | {name} -> {saved}")

        if needs_sync:
            ctypes.pythonapi.PyFrame_LocalsToFast(
                ctypes.py_object(frame),
                ctypes.c_int(0)
            )

        # ── CAPTURE scalars ───────────────────────────────────────────
        if found_scalars:
            with self._lock:
                self._captured.update(found_scalars)
                self.logger.debug(f"[EXTRACT] | scalars updated | {found_scalars}")

        # ── DISCOVER objects ──────────────────────────────────────────────
        if found_objects:
            with self._lock:
                for name, obj in found_objects.items():
                    self._objects[name] = obj
                    self.logger.info(f"[EXTRACT] | object discovered | name={name} | type={type(obj).__name__}")
                    if name in self._pending_restore:
                        saved = self._pending_restore.pop(name)
                        try:
                            obj.load_state_dict(saved)
                            self.logger.info(f"[RESTORE] | load_state_dict applied | {name}")
                        except Exception as e:
                            self.logger.error(f"[RESTORE] | load_state_dict failed | name={name} | reason={e}")
                            raise RuntimeError(f"[RESTORE] | load_state_dict failed | name={name} | reason={e}") from e

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
