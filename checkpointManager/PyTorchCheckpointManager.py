"""
checkpoint/PyTorchCheckpointManager.py

Concrete PyTorch implementation of CheckpointManager.
Uses torch. Save / torch. Load.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from checkpointManager.CheckpointManager import CheckpointManager


# ---------------------------------------------------------------------------
# PyTorch implementation
# ---------------------------------------------------------------------------

_CKPT_RE = re.compile(r"^checkpoint_(\d+)\.pt$")


class PyTorchCheckpointManager(CheckpointManager):
    """
    Saves and loads PyTorch checkpoints with atomic writes and optional
    checkpoint rotation.

    File layout inside *checkpoint_dir*::

        checkpoint_dir/
            checkpoint_00000100.pt
            checkpoint_00000200.pt
            ...

    Parameters
    ----------
    checkpoint_dir:
        Directory where checkpoint files are stored (created on demand).
    max_to_keep:
        Maximum number of checkpoint files to retain.  Oldest files are
        deleted when the limit is exceeded.  ``None`` means keep all.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        max_to_keep: Optional[int] = None,
    ) -> None:
        self._dir = Path(checkpoint_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep

    # ------------------------------------------------------------------
    # Public API — high-level helpers (called by RunnerScript / AC)
    # ------------------------------------------------------------------

    def save(
        self,
        step: int,
        model: nn.Module,
        optimizer: optim.Optimizer,
        **kwargs: Any,
    ) -> str:
        """
        Save a full training checkpoint.

        Parameters
        ----------
        step:
            Global training step / iteration number used in the filename.
        model:
            PyTorch module whose ``state_dict`` will be saved.
        optimizer:
            Optimizer whose ``state_dict`` will be saved.
        **kwargs:
            Any extra metadata (e.g. ``epoch``, ``loss``, ``scheduler``
            state dict) appended verbatim to the checkpoint payload.

        Returns
        -------
        str
            Absolute path of the saved checkpoint file.
        """
        payload: Dict[str, Any] = {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            **kwargs,
        }
        path = self._checkpoint_path(step)
        self._atomic_save(payload, path)

        if self.max_to_keep is not None:
            self._prune()

        return str(path)

    def load(
        self,
        model: nn.Module,
        optimizer: Optional[optim.Optimizer] = None,
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load a checkpoint and restore model (and optionally optimizer) state.

        Parameters
        ----------
        model:
            PyTorch module to restore.
        optimizer:
            If provided, its state is also restored.
        checkpoint_path:
            Explicit path to a ``.pt`` file.  When ``None`` the most recent
            checkpoint in *checkpoint_dir* is used automatically.

        Returns
        -------
        dict
            The full checkpoint payload (always contains ``"step"``; may
            contain any extra keys that were passed to :meth:`save`).

        Raises
        ------
        FileNotFoundError
            If *checkpoint_path* is given but does not exist.
        RuntimeError
            If *checkpoint_path* is ``None`` and no checkpoints are found.
        """
        if checkpoint_path is not None:
            path = Path(checkpoint_path)
            if not path.is_file():
                raise FileNotFoundError(
                    f"Checkpoint file not found: {checkpoint_path}"
                )
        else:
            latest = self._get_latest_checkpoint()
            if latest is None:
                raise RuntimeError(
                    f"No checkpoints found in directory: {self._dir}"
                )
            path = Path(latest)

        # map_location keeps the code device-agnostic
        payload: Dict[str, Any] = torch.load(
            path,
            map_location=lambda storage, _: storage,
            weights_only=False,
        )

        model.load_state_dict(payload["model_state_dict"])

        if optimizer is not None and "optimizer_state_dict" in payload:
            optimizer.load_state_dict(payload["optimizer_state_dict"])

        # Return everything except the bulky state dicts so callers get clean
        # metadata (step, epoch, loss …).  The full payload is also returned
        # so nothing is lost — callers can ignore what they don't need.
        return payload

    # ------------------------------------------------------------------
    # CheckpointManager abstract interface
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        state: Dict[str, Any],
        save_dir: str,
    ) -> str:
        """
        Persist an arbitrary *state* dict to *save_dir*.

        This satisfies the abstract interface used by the AutonomousController.
        The step is read from ``state["step"]`` (defaults to 0).

        Parameters
        ----------
        state:
            Snapshot dict produced by a StateTracker.  Expected keys:
            ``"step"``, ``"model_state_dict"``, ``"optimizer_state_dict"``.
        save_dir:
            Directory in which to write the file.  Overrides the instance
            *checkpoint_dir* for this single call.
        """
        target_dir = Path(save_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        step: int = state.get("step", 0)
        path = target_dir / f"checkpoint_{step:08d}.pt"
        self._atomic_save(state, path)

        if self.max_to_keep is not None:
            self._prune(directory=target_dir)
        return str(path)

    def load_checkpoint(
        self,
        save_dir: str,
    ) -> Dict[str, Any]:
        """
        Recover the most recent checkpoint from *save_dir*.

        Parameters
        ----------
        save_dir:
            Directory to search for checkpoint files.

        Returns
        -------
        dict
            Raw checkpoint payload.

        Raises
        ------
        RuntimeError
            If *save_dir* contains no recognised checkpoint files.
        """
        directory = Path(save_dir)
        latest = self._get_latest_checkpoint(directory=directory)
        if latest is None:
            raise RuntimeError(
                f"No checkpoints found in directory: {directory}"
            )
        payload: Dict[str, Any] = torch.load(
            latest,
            map_location=lambda storage, _: storage,
            weights_only=False,
        )
        return payload

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_latest_checkpoint(
        self,
        directory: Optional[Path] = None,
    ) -> Optional[str]:
        """
        Return the path of the checkpoint with the highest step number,
        or ``None`` if the directory contains no checkpoint files.

        Parameters
        ----------
        directory:
            Directory to search.  Defaults to the instance *checkpoint_dir*.
        """
        files = self._list_checkpoints(directory or self._dir)
        return str(files[-1]) if files else None

    def _list_checkpoints(self, directory: Path) -> List[Path]:
        """
        Return checkpoint files in *directory* sorted by step (ascending).
        Only files whose names match ``checkpoint_<digits>.pt`` are included.
        """
        found: List[tuple[int, Path]] = []
        for p in directory.iterdir():
            m = _CKPT_RE.match(p.name)
            if m:
                found.append((int(m.group(1)), p))
        found.sort(key=lambda t: t[0])
        return [p for _, p in found]

    def _checkpoint_path(self, step: int) -> Path:
        return self._dir / f"checkpoint_{step:08d}.pt"

    def _atomic_save(self, payload: Dict[str, Any], path: Path) -> None:
        """
        Write *payload* to *path* atomically via a sibling ``.tmp`` file.

        Using ``os.replace`` guarantees that readers never see a partially
        written file, even if the process is killed mid-write.
        """
        tmp_fd, tmp_name = tempfile.mkstemp(
            dir=path.parent, suffix=".tmp"
        )
        try:
            os.close(tmp_fd)
            torch.save(payload, tmp_name)
            os.replace(tmp_name, path)
        except Exception:
            # Clean up the temp file if anything goes wrong
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    def _prune(self, directory: Optional[Path] = None) -> None:
        """
        Delete the oldest checkpoints so that at most ``max_to_keep`` remain.

        Parameters
        ----------
        directory:
            Directory to prune.  Defaults to the instance *checkpoint_dir*.
        """
        if self.max_to_keep is None:
            return
        files = self._list_checkpoints(directory or self._dir)
        for old in files[: -self.max_to_keep]:
            try:
                old.unlink()
            except OSError:
                pass  # already deleted by a concurrent process — harmless