"""
checkpointManager/KerasCheckpointManager.py

Concrete Keras / TensorFlow implementation of CheckpointManager.

Uses ``tf.train.Checkpoint`` to save and restore model weights and optimizer
slots — the only TF-native mechanism that reliably serialises optimizer state
(momentum, variance, etc.) across sessions.

Checkpoint layout inside *checkpoint_dir*::

    checkpoint_dir/
        checkpoint_00000100/
            weights.index
            weights.data-00000-of-00001   ← tf.train.Checkpoint binary shards
            metadata.json                 ← step + any kwargs
        checkpoint_00000200/
            ...
        session_info.json                 ← written by save_session_info()
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Lazy TensorFlow import — avoids import-time cost when not using Keras
# ---------------------------------------------------------------------------

def _tf():
    """Return the ``tensorflow`` module, raising a clear error if absent."""
    try:
        import tensorflow as tf  # noqa: PLC0415
        return tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for KerasCheckpointManager. "
            "Install it with:  pip install tensorflow"
        ) from exc


# ---------------------------------------------------------------------------
# Import the project's abstract base
# ---------------------------------------------------------------------------

from checkpointManager.CheckpointManager import CheckpointManager

logger = logging.getLogger(__name__)

# Matches sub-directory names produced by this manager
_CKPT_DIR_RE = re.compile(r"^checkpoint_(\d+)$")

# Keys in a state snapshot that hold live TF objects — not JSON-serialisable
_TF_OBJECT_KEYS = {"model", "optimizer"}


# ---------------------------------------------------------------------------
# KerasCheckpointManager
# ---------------------------------------------------------------------------

class KerasCheckpointManager(CheckpointManager):
    """
    Saves and loads Keras / TensorFlow checkpoints.

    Each checkpoint lives in its own numbered sub-directory so that the
    TF binary shards and the JSON metadata file are never mixed with files
    from other steps::

        checkpoint_dir/
            checkpoint_00000100/
                weights.index
                weights.data-00000-of-00001
                metadata.json
            checkpoint_00000200/
                ...

    Parameters
    ----------
    checkpoint_dir:
        Root directory.  Created automatically if absent.
    max_to_keep:
        Maximum sub-directories to keep.  ``None`` keeps all.
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
    # High-level helpers — called directly from training loops
    # ------------------------------------------------------------------

    def save(
        self,
        step: int,
        model: Any,       # tf.keras.Model
        optimizer: Any,   # tf.keras.optimizers.Optimizer
        **kwargs: Any,
    ) -> str:
        """
        Save a full training checkpoint.

        Parameters
        ----------
        step:
            Global training step used in the directory name.
        model:
            A compiled ``tf.keras.Model``.
        optimizer:
            The optimizer whose slots (momentum, variance …) are saved.
            Pass ``None`` to skip optimizer state.
        **kwargs:
            Any JSON-serialisable metadata (``epoch``, ``loss``, …).

        Returns
        -------
        str
            Absolute path to the checkpoint sub-directory.
        """
        ckpt_dir = self._checkpoint_subdir(step)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # ── 1. Save weights + optimizer via tf.train.Checkpoint ───────
        self._save_tf_checkpoint(model, optimizer, ckpt_dir)

        # ── 2. Write JSON metadata ────────────────────────────────────
        metadata: Dict[str, Any] = {"step": step, **kwargs}
        self._atomic_json_save(metadata, ckpt_dir / "metadata.json")

        # ── 3. Rotate old checkpoints ─────────────────────────────────
        if self.max_to_keep is not None:
            self._prune()

        logger.info("[KerasCheckpointManager] Saved -> %s", ckpt_dir)
        return str(ckpt_dir)

    def load(
        self,
        model: Any,                        # tf.keras.Model
        optimizer: Optional[Any] = None,   # tf.keras.optimizers.Optimizer
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Load a checkpoint and restore model (and optionally optimizer) state.

        Parameters
        ----------
        model:
            Keras model to restore into.
        optimizer:
            If provided, its slots are also restored.  The optimizer must
            have been built (called at least once) before loading.
        checkpoint_path:
            Explicit path to a checkpoint *sub-directory*.
            When ``None`` the latest checkpoint is used automatically.

        Returns
        -------
        dict
            The metadata dict (always contains ``"step"``).

        Raises
        ------
        FileNotFoundError
            If *checkpoint_path* is given but does not exist.
        RuntimeError
            If no checkpoints are found in *checkpoint_dir*.
        """
        if checkpoint_path is not None:
            ckpt_dir = Path(checkpoint_path)
            if not ckpt_dir.is_dir():
                raise FileNotFoundError(
                    f"Checkpoint directory not found: {checkpoint_path}"
                )
        else:
            latest = self._get_latest_checkpoint()
            if latest is None:
                raise RuntimeError(
                    f"No checkpoints found in: {self._dir}"
                )
            ckpt_dir = Path(latest)

        # ── 1. Restore weights + optimizer ────────────────────────────
        self._restore_tf_checkpoint(model, optimizer, ckpt_dir)

        # ── 2. Read metadata ──────────────────────────────────────────
        meta_path = ckpt_dir / "metadata.json"
        metadata: Dict[str, Any] = {}
        if meta_path.exists():
            with meta_path.open("r", encoding="utf-8") as fh:
                metadata = json.load(fh)

        logger.info("[KerasCheckpointManager] Loaded <- %s", ckpt_dir)
        return metadata

    # ------------------------------------------------------------------
    # CheckpointManager abstract interface — called by AutonomousController
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        state: Dict[str, Any],
        save_dir: str,
    ) -> str:
        """
        Persist a raw state snapshot dict to *save_dir*.

        Called by the ``AutonomousController`` with the dict produced by
        ``StateTracker.snapshot()``.

        Expected keys (all optional — missing ones are skipped gracefully):

        * ``"step"`` / ``"global_step"`` — used in the directory name.
        * ``"model"``    — live ``tf.keras.Model`` instance.
        * ``"optimizer"`` — live optimizer instance.
        * Any other JSON-serialisable key is written to ``metadata.json``.

        Parameters
        ----------
        state:
            Snapshot dict from the StateTracker.
        save_dir:
            Destination root directory.

        Returns
        -------
        str
            Absolute path to the checkpoint sub-directory.
        """
        target_root = Path(save_dir)
        target_root.mkdir(parents=True, exist_ok=True)

        requested_step: int = int(state.get("step", state.get("global_step", 0)))
        model     = state.get("model")
        optimizer = state.get("optimizer")

        # Keep checkpoint directory numbering monotonic within save_dir.
        # This prevents older/lower user step values from being treated as "latest".
        existing = self._list_checkpoints(target_root)
        next_step = int(existing[-1].name.split("_")[1]) + 1 if existing else 0
        step = max(requested_step, next_step)

        ckpt_dir = target_root / f"checkpoint_{step:08d}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save TF objects if present
        if model is not None:
            self._save_tf_checkpoint(model, optimizer, ckpt_dir)

        # Metadata = everything that is JSON-serialisable
        metadata: Dict[str, Any] = {
            k: v for k, v in state.items()
            if k not in _TF_OBJECT_KEYS
        }
        metadata["step"] = step
        self._atomic_json_save(metadata, ckpt_dir / "metadata.json")

        self.save_session_info(save_dir, checkpoint_path=str(ckpt_dir))

        if self.max_to_keep is not None:
            self._prune(directory=target_root)

        logger.info("[KerasCheckpointManager] Saved -> %s", ckpt_dir)
        return str(ckpt_dir)

    def load_checkpoint(
        self,
        save_dir: str,
    ) -> Dict[str, Any]:
        """
        Recover the latest checkpoint metadata from *save_dir*.

        This returns only the JSON metadata.  To restore model weights
        call :meth:`load` directly.

        Parameters
        ----------
        save_dir:
            Root directory to search.

        Returns
        -------
        dict
            Metadata payload of the latest checkpoint.

        Raises
        ------
        RuntimeError
            If no checkpoints are found.
        FileNotFoundError
            If the metadata file is missing from the latest checkpoint.
        """
        directory = Path(save_dir)
        latest = self._get_latest_checkpoint(directory=directory)
        if latest is None:
            raise RuntimeError(
                f"No checkpoints found in directory: {directory}"
            )
        ckpt_dir = Path(latest)
        meta_path = ckpt_dir / "metadata.json"

        if not meta_path.exists():
            raise FileNotFoundError(
                f"Metadata file missing from checkpoint: {meta_path}"
            )
        with meta_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_latest_checkpoint(
        self,
        directory: Optional[Path] = None,
    ) -> Optional[str]:
        """
        Return the path of the checkpoint sub-directory with the highest
        step number, or ``None`` if none exist.
        """
        entries = self._list_checkpoints(directory or self._dir)
        return str(entries[-1]) if entries else None

    def _list_checkpoints(self, directory: Path) -> List[Path]:
        """
        Return checkpoint sub-directories sorted by step number (ascending).
        Only directories matching ``checkpoint_<digits>`` are included.
        """
        if not directory.is_dir():
            return []
        found: List[Tuple[int, Path]] = []
        for entry in directory.iterdir():
            if not entry.is_dir():
                continue
            m = _CKPT_DIR_RE.match(entry.name)
            if m:
                found.append((int(m.group(1)), entry))
        found.sort(key=lambda t: t[0])
        return [p for _, p in found]

    def _checkpoint_subdir(self, step: int) -> Path:
        return self._dir / f"checkpoint_{step:08d}"

    # ------------------------------------------------------------------
    # TF Checkpoint read / write
    # ------------------------------------------------------------------

    def _save_tf_checkpoint(
        self,
        model: Any,
        optimizer: Optional[Any],
        ckpt_dir: Path,
    ) -> None:
        """
        Write model weights (and optionally optimizer slots) into *ckpt_dir*
        using ``tf.train.Checkpoint``.

        ``tf.train.Checkpoint`` writes several binary shard files plus an
        ``.index`` file.  We name the prefix ``weights`` so the resulting
        files are ``weights.index``, ``weights.data-00000-of-00001``, etc.

        The save is done to a sibling ``_tmp`` directory first, then the
        whole directory is atomically renamed — guards against partial writes
        if the process is killed mid-save.
        """
        tf = _tf()

        ckpt_kwargs: Dict[str, Any] = {"model": model}
        if optimizer is not None:
            ckpt_kwargs["optimizer"] = optimizer

        checkpoint = tf.train.Checkpoint(**ckpt_kwargs)

        # Write to a temp sibling dir, then rename atomically
        tmp_dir = ckpt_dir.parent / (ckpt_dir.name + "_tmp")
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

        try:
            checkpoint.write(str(tmp_dir / "weights"))

            # Move each file from tmp_dir into the real ckpt_dir
            for src in tmp_dir.iterdir():
                dst = ckpt_dir / src.name
                shutil.move(str(src), str(dst))
        finally:
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)

    def _restore_tf_checkpoint(
        self,
        model: Any,
        optimizer: Optional[Any],
        ckpt_dir: Path,
    ) -> None:
        """
        Restore model weights (and optionally optimizer slots) from *ckpt_dir*.

        Uses ``tf.train.Checkpoint.restore(...).expect_partial()`` so that
        the restore does not raise if the optimizer was not saved (e.g. when
        the caller passes ``optimizer=None`` but slots exist in the file).

        Parameters
        ----------
        model:
            Must be built before calling this (i.e. called on at least one
            input batch) so that variable shapes are known.
        optimizer:
            If ``None``, optimizer slots are simply ignored.
        ckpt_dir:
            Sub-directory containing ``weights.index`` and the data shards.
        """
        tf = _tf()

        weights_prefix = str(ckpt_dir / "weights")

        # Verify the checkpoint files actually exist before trying to load
        index_file = Path(weights_prefix + ".index")
        if not index_file.exists():
            raise FileNotFoundError(
                f"TF checkpoint index not found: {index_file}. "
                "The checkpoint directory may be corrupt or contain only metadata."
            )

        ckpt_kwargs: Dict[str, Any] = {"model": model}
        if optimizer is not None:
            ckpt_kwargs["optimizer"] = optimizer

        checkpoint = tf.train.Checkpoint(**ckpt_kwargs)

        # expect_partial() suppresses warnings about unmatched variables when
        # the optimizer is absent or the model architecture differs slightly.
        status = checkpoint.restore(weights_prefix)
        status.expect_partial()

    # ------------------------------------------------------------------
    # Metadata and rotation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _atomic_json_save(data: Dict[str, Any], path: Path) -> None:
        """Write *data* as JSON to *path* atomically via a sibling tmp file."""
        tmp_fd, tmp_name = tempfile.mkstemp(
            dir=path.parent, suffix=".json.tmp"
        )
        try:
            os.close(tmp_fd)
            with open(tmp_name, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, default=_json_default)
            os.replace(tmp_name, path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    def _prune(self, directory: Optional[Path] = None) -> None:
        """
        Delete the oldest checkpoint sub-directories so that at most
        ``max_to_keep`` remain.
        """
        if self.max_to_keep is None:
            return
        entries = self._list_checkpoints(directory or self._dir)
        for old in entries[: -self.max_to_keep]:
            try:
                shutil.rmtree(old)
                logger.debug("[KerasCheckpointManager] Pruned: %s", old)
            except OSError:
                pass  # already removed — harmless


# ---------------------------------------------------------------------------
# JSON serialisation helpers
# ---------------------------------------------------------------------------

def _json_default(obj: Any) -> Any:
    """Fallback serialiser for numpy scalars and arrays in metadata dicts."""
    try:
        import numpy as np  # noqa: PLC0415
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:
        pass
    raise TypeError(
        f"Object of type {type(obj).__name__} is not JSON serialisable"
    )
