from __future__ import annotations
import json
import logging
import os
import shutil
import tempfile
import re
from pathlib import Path
from typing import Any, Dict
from .CheckpointManager import CheckpointManager

logger = logging.getLogger(__name__)


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
        versions = self._scan_versions(save_dir)
        next_version = (versions[-1] + 1) if versions else 1
        logger.info("[KerasCKPT] Saving as version v%d -> %s", next_version, save_dir)

        # ── Write everything to a temp dir, then swap atomically ──────
        parent_dir = os.path.abspath(save_dir)
        os.makedirs(parent_dir, exist_ok=True)

        with tempfile.TemporaryDirectory(dir=parent_dir, prefix=".tmp_ckpt_") as tmp_dir:

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

                    # Save both model + optimizer together (IMPORTANT FIX)
                    tf_ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
                    tf_ckpt.write(os.path.join(opt_dir, "ckpt"))

                    logger.info("[KerasCKPT] Model + Optimizer saved -> %s", opt_dir)
                except Exception as e:
                    logger.warning("[KerasCKPT] Could not save optimizer: %s", e)

            # 3. Build and save metadata (with version + model checksum)
            skip_keys = {"model", "optimizer"}
            metadata: Dict[str, Any] = {
                k: v for k, v in state.items()
                if k not in skip_keys and self._is_json_serializable(v)
            }
            # ensure filesystem flush safety
            if hasattr(model, "save"):
                pass  # no-op but keeps structure clear

            opt_config = optimizer.get_config() if optimizer is not None else None
            metadata.update({
                "checkpoint_version": next_version,
                "model_filename": self.model_filename,
                "optimizer_config": opt_config,
                "model_checksum": self._checksum(model_path),
            })

            meta_path = os.path.join(version_subdir, self.METADATA_FILE)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            logger.info("[KerasCKPT] Metadata saved -> %s", meta_path)

            # 4. Session info
            self.save_session_info(version_subdir)

            # 5. Atomic swap (safe rename strategy)
            final_version_dir = os.path.join(save_dir, f"v{next_version}")

            #Atomic replace
            os.replace(version_subdir, final_version_dir)
        logger.info("[KerasCKPT] Checkpoint complete -> %s", save_dir)
        return os.path.join(save_dir, f"v{next_version}")

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
        versions = self._scan_versions(save_dir)
        if not versions:
            logger.error("[KerasCKPT] No previous checkpoints")
            raise FileNotFoundError("No checkpoints found")

        latest_version = versions[-1]

        # iterate from newest → oldest once (no extra disk scans)
        selected_version = None
        for v in reversed(versions):
            version_subdir = os.path.join(save_dir, f"v{v}")
            if self._is_valid_checkpoint(version_subdir):
                selected_version = v
                break

        if selected_version is None:
            raise FileNotFoundError("No valid checkpoints found (all versions are corrupted).")

        if selected_version != latest_version:
            logger.warning(
                "[KerasCKPT] Falling back from v%d to last valid v%d",latest_version, selected_version)

        logger.info("[KerasCKPT] Loading checkpoint v%d (requested latest v%d)",selected_version, latest_version)

        # 1. Load metadata first (so we can read model_filename + checksum)
        version_subdir = os.path.join(save_dir, f"v{selected_version}")

        meta_path = os.path.join(version_subdir, self.METADATA_FILE)
        with open(meta_path, "r", encoding="utf-8") as fh:
            metadata = json.load(fh)

        # ── Validate checksum before loading model ───────────────────────
        expected_checksum = metadata.get("model_checksum")
        if expected_checksum is not None:
            actual_checksum = self._checksum(
                os.path.join(version_subdir, metadata.get("model_filename", self.model_filename))
            )
            if actual_checksum != expected_checksum:
                raise ValueError(
                    f"[KerasCKPT] Checksum mismatch for v{selected_version}. "
                    "Checkpoint may be corrupted."
                )

        model_path = os.path.join(version_subdir, metadata.get("model_filename", self.model_filename))
        model = tf.keras.models.load_model(model_path, compile=False)

        state = dict(metadata)
        state["model"] = model
        logger.info("[KerasCKPT] Metadata loaded <- %s", meta_path)

        opt_dir = os.path.join(version_subdir, self.OPTIMIZER_DIR)
        ckpt_path = os.path.join(opt_dir, "ckpt")

        if os.path.exists(opt_dir):
            opt_config = metadata.get("optimizer_config")
            if opt_config is not None:
                optimizer = tf.keras.optimizers.Adam.from_config(opt_config)
            else:
                optimizer = tf.keras.optimizers.Adam()
            ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
            status = ckpt.restore(ckpt_path)
            status.assert_existing_objects_matched()
            state["optimizer"] = optimizer
        else:
            state["optimizer"] = None

        state["session_info"] = self.load_session_info(version_subdir)

        return state
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _scan_versions(self, save_dir: str) -> list[int]:
        """
        Scan save_dir once and return sorted list of available versions.
        """
        path = Path(save_dir)
        if not path.exists():
            return []
        versions = []
        for p in path.iterdir():
            if p.is_dir():
                m = re.match(r"v(\d+)", p.name)
                if m:
                    versions.append(int(m.group(1)))

        return sorted(versions)

    def _is_json_serializable(self, value: Any) -> bool:
        """Return True if *value* can be written directly to JSON."""
        try:
            json.dumps(value)
            return True
        except (TypeError, ValueError):
            return False

    def _checksum(self, path: str) -> str:
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

    def _is_valid_checkpoint(self, version_dir: str) -> bool:
        meta = os.path.join(version_dir, self.METADATA_FILE)
        model = os.path.join(version_dir, self.model_filename)

        if not (os.path.exists(meta) and os.path.exists(model)):
            return False
        try:
            with open(meta, "r", encoding="utf-8") as f:
                json.load(f)
        except Exception:
            return False

        return True

