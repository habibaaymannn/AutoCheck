from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from typing import Any, Dict

from .CheckpointManager import CheckpointManager

logger = logging.getLogger(__name__)


# No module-level CHECKPOINT_VERSION — versions are derived from the filesystem.


class KerasCheckpointManager(CheckpointManager):
    """
    Checkpoint manager for Keras / TensorFlow models.

    Inherits session info load/save from CheckpointManager and
    implements save_checkpoint / load_checkpoint using the native
    Keras SavedModel format (.keras file) together with a companion
    JSON file that carries all scalar metadata (epoch, step, ...).

    Directory layout produced by save_checkpoint():
        <save_dir>/
            v{checkpoint_version}/
                model.keras          <- full Keras model (weights + architecture + compile state)
                optimizer/           <- tf.train.Checkpoint directory (weights + step)
                metadata.json        <- every non-model key from the state dict + version + checksum
                session_info.json    <- written by CheckpointManager.save_session_info()

    Each call to save_checkpoint() creates a new v{n+1}/ subdirectory,
    preserving all previous versions.  load_checkpoint() always loads the
    highest-numbered version found on disk.

    Writes are atomic: everything goes to a temp dir first, then the
    temp dir is swapped into place so a crash mid-save never leaves a
    partial checkpoint.

    Parameters
    ----------
    model_filename : str
        Name of the Keras model file inside the versioned subdirectory.
        Defaults to "model.keras".
    """

    MODEL_FILE = "model.keras"  # public class constant for external reference
    OPTIMIZER_DIR = "optimizer"
    METADATA_FILE = "metadata.json"

    def __init__(self, model_filename: str = "model.keras") -> None:
        self.model_filename = model_filename

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_checkpoint(self, state: Dict[str, Any], save_dir: str) -> str:
        """
        Persist a full training snapshot to *save_dir* atomically.

        Each call appends a new versioned subdirectory (v1/, v2/, ...).
        The version number is derived from what already exists on disk:
            - save_dir is empty or missing  -> writes to v1/
            - highest existing version is n -> writes to v{n+1}/

        Expected keys in *state*
        ------------------------
        model          : tf.keras.Model  — the live Keras model object
        optimizer      : (optional) tf.keras.optimizers.Optimizer
        epoch          : int
        global_step    : int  (optional)
        batch_idx      : int  (optional)
        ...any extra scalar / dict fields are stored in metadata.json

        Returns
        -------
        str  — absolute path to the versioned subdirectory that was written.
        """
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is required for KerasCheckpointManager. "
                "Install it with: pip install tensorflow"
            ) from exc

        # ── Guard: save_dir must not already exist as a file ─────────
        if os.path.isfile(save_dir):
            raise ValueError(
                f"save_dir '{save_dir}' already exists as a file. "
                "Provide a directory path instead."
            )

        # ── Validate model ────────────────────────────────────────────
        model: tf.keras.Model = state.get("model")
        if model is None:
            raise KeyError("'model' key is missing from state — cannot save Keras checkpoint.")
        if not isinstance(model, tf.keras.Model):
            raise TypeError(
                f"'model' must be a tf.keras.Model instance, got {type(model).__name__}."
            )

        # ── Determine next version from what already exists on disk ───
        next_version = self._resolve_next_version(save_dir)
        logger.info("[KerasCKPT] Saving as version v%d -> %s", next_version, save_dir)

        # ── Write everything to a temp dir, then swap atomically ──────
        parent_dir = os.path.dirname(os.path.abspath(save_dir)) or "."
        os.makedirs(parent_dir, exist_ok=True)

        with tempfile.TemporaryDirectory(dir=parent_dir, prefix=".tmp_ckpt_") as tmp_dir:

            # Preserve any existing vN/ subdirs so older versions survive the swap
            if os.path.isdir(save_dir):
                for entry in os.scandir(save_dir):
                    if entry.is_dir() and entry.name.startswith("v"):
                        shutil.copytree(entry.path, os.path.join(tmp_dir, entry.name))

            # All new artefacts live under the next versioned subdirectory
            version_subdir = os.path.join(tmp_dir, f"v{next_version}")
            os.makedirs(version_subdir, exist_ok=True)

            # 1. Save the Keras model
            model_path = os.path.join(version_subdir, self.model_filename)
            model.save(model_path)
            logger.info("[KerasCKPT] Model saved -> %s", model_path)

            # 2. Save optimizer via tf.train.Checkpoint (version-stable)
            optimizer = state.get("optimizer")
            if optimizer is not None:
                try:
                    opt_dir = os.path.join(version_subdir, self.OPTIMIZER_DIR)
                    os.makedirs(opt_dir, exist_ok=True)
                    tf_ckpt = tf.train.Checkpoint(optimizer=optimizer)
                    tf_ckpt.write(os.path.join(opt_dir, "ckpt"))
                    logger.info("[KerasCKPT] Optimizer saved -> %s", opt_dir)
                except Exception as e:
                    logger.warning("[KerasCKPT] Could not save optimizer: %s", e)

            # 3. Build and save metadata (with version + model checksum)
            skip_keys = {"model", "optimizer"}
            metadata: Dict[str, Any] = {
                k: v for k, v in state.items()
                if k not in skip_keys and self._is_json_serialisable(v)
            }
            metadata["checkpoint_version"] = next_version
            metadata["model_filename"] = self.model_filename
            metadata["model_checksum"] = self._file_checksum(model_path)

            meta_path = os.path.join(version_subdir, self.METADATA_FILE)
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump(metadata, fh, indent=2)
            logger.info("[KerasCKPT] Metadata saved -> %s", meta_path)

            # 4. Session info
            self.save_session_info(version_subdir)

            # 5. Atomic swap: replace save_dir with the fully prepared tmp dir
            if os.path.isdir(save_dir):
                shutil.rmtree(save_dir)
            shutil.copytree(tmp_dir, save_dir)

        logger.info("[KerasCKPT] Checkpoint complete -> %s", save_dir)
        return os.path.abspath(os.path.join(save_dir, f"v{next_version}"))

    def load_checkpoint(self, save_dir: str) -> Dict[str, Any]:
        """
        Load the latest checkpoint from *save_dir*.

        The latest version is determined automatically by scanning for the
        highest-numbered v{n}/ subdirectory — no version number needs to be
        supplied by the caller.

        Returns
        -------
        dict with keys:
            model              : tf.keras.Model   (fully restored)
            optimizer_ckpt_dir : str | None       (path to optimizer ckpt prefix —
                                                   call tf.train.Checkpoint(optimizer=opt)
                                                   .read(path) after the first training step)
            checkpoint_version : int              (the version that was loaded)
            epoch, global_step, batch_idx, ...    (all scalars from metadata.json)
            session_info       : dict | None      (from session_info.json)
        """
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is required for KerasCheckpointManager. "
                "Install it with: pip install tensorflow"
            ) from exc

        if not os.path.isdir(save_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {save_dir}")

        # Auto-detect the latest versioned subdirectory (e.g. save_dir/v3/)
        latest_version = self._resolve_latest_version(save_dir)
        version_subdir = os.path.join(save_dir, f"v{latest_version}")
        logger.info("[KerasCKPT] Loading latest checkpoint v%d <- %s", latest_version, save_dir)

        state: Dict[str, Any] = {}

        # 1. Load metadata first (so we can read model_filename + checksum)
        meta_path = os.path.join(version_subdir, self.METADATA_FILE)
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"metadata.json not found in '{version_subdir}'. "
                "The checkpoint may be corrupt."
            )
        with open(meta_path, "r", encoding="utf-8") as fh:
            metadata = json.load(fh)

        state.update(metadata)
        logger.info("[KerasCKPT] Metadata loaded <- %s", meta_path)

        # 2. Load and verify the Keras model
        model_filename = metadata.get("model_filename", self.model_filename)
        model_path = os.path.join(version_subdir, model_filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        saved_checksum = metadata.get("model_checksum")
        if saved_checksum is not None:
            actual_checksum = self._file_checksum(model_path)
            if actual_checksum != saved_checksum:
                raise ValueError(
                    f"Model file checksum mismatch for '{model_path}'. "
                    "The file may be corrupt."
                )

        state["model"] = tf.keras.models.load_model(model_path)
        logger.info("[KerasCKPT] Model loaded <- %s", model_path)

        # 3. Note optimizer checkpoint path (caller restores after first step)
        opt_dir = os.path.join(version_subdir, self.OPTIMIZER_DIR)
        if os.path.isdir(opt_dir):
            state["optimizer_ckpt_dir"] = os.path.join(opt_dir, "ckpt")
            logger.info("[KerasCKPT] Optimizer checkpoint found <- %s", opt_dir)
        else:
            state["optimizer_ckpt_dir"] = None
            logger.warning("[KerasCKPT] No optimizer checkpoint found in %s", version_subdir)

        # 4. Session info
        state["session_info"] = self.load_session_info(version_subdir)

        logger.info("[KerasCKPT] Checkpoint loaded <- %s", save_dir)
        return state

    # _is_json_serialisable, _file_checksum, _resolve_next_version,
    # and _resolve_latest_version are all inherited from CheckpointManager.

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_next_version(save_dir: str) -> int:
        """
        Scan *save_dir* for existing v{n}/ subdirectories and return the
        next version number to write.

        - Directory is empty (or does not exist yet) → returns 1
        - Highest existing version is n                → returns n + 1
        """
        return CheckpointManager._latest_version(save_dir) + 1

    @staticmethod
    def _resolve_latest_version(save_dir: str) -> int:
        """
        Scan *save_dir* for existing v{n}/ subdirectories and return the
        highest version number found.

        Raises FileNotFoundError if no versioned subdirectory exists.
        """
        latest = CheckpointManager._latest_version(save_dir)
        if latest == 0:
            raise FileNotFoundError(
                f"No versioned checkpoint subdirectory (v1, v2, …) found in '{save_dir}'."
            )
        return latest

    @staticmethod
    def _latest_version(save_dir: str) -> int:
        """Return the highest v{n} version found in *save_dir*, or 0 if none."""
        if not os.path.isdir(save_dir):
            return 0
        highest = 0
        for entry in os.scandir(save_dir):
            if entry.is_dir() and entry.name.startswith("v"):
                try:
                    n = int(entry.name[1:])
                    if n > highest:
                        highest = n
                except ValueError:
                    pass  # ignore directories like "valid", "viz", etc.
        return highest

    @staticmethod
    def _is_json_serialisable(value: Any) -> bool:
        """Return True if *value* can be written directly to JSON."""
        try:
            json.dumps(value)
            return True
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _file_checksum(path: str) -> str:
        """
        Return a SHA-256 hex digest for the file (or directory) at *path*.

        For directories every file is hashed in sorted walk order so the
        digest is stable across platforms.
        """
        import hashlib

        sha = hashlib.sha256()
        if os.path.isfile(path):
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(65536), b""):
                    sha.update(chunk)
        elif os.path.isdir(path):
            for root, _, files in os.walk(path):
                for fname in sorted(files):
                    fpath = os.path.join(root, fname)
                    with open(fpath, "rb") as fh:
                        for chunk in iter(lambda: fh.read(65536), b""):
                            sha.update(chunk)
        else:
            raise FileNotFoundError(f"Cannot checksum — path not found: {path}")
        return sha.hexdigest()
