from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .CheckpointManager import CheckpointManager

logger = logging.getLogger(__name__)


class KerasCheckpointManager(CheckpointManager):
    """
    Checkpoint manager for Keras / TensorFlow models.

    Directory layout:
        <save_dir>/
            v{n}/
                model.keras             <- full Keras model (weights + architecture + compile state)
                optimizer_state.npy     <- optimizer weights as numpy array (optional)
                metadata.json           <- all scalar/dict state fields + checksum
                session_info.json       <- written by CheckpointManager.save_session_info()

    Saves are atomic (temp dir → rename). Each call creates a new version;
    load always picks the highest valid version (checksum-verified).
    """

    MODEL_FILE     = "model.keras"
    OPTIMIZER_FILE = "optimizer_state.npy"
    METADATA_FILE  = "metadata.json"
    VERSION_RE     = re.compile(r"^v(\d+)$")

    def __init__(self, model_filename: str = "model.keras") -> None:
        super().__init__()
        self.model_filename = model_filename

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_checkpoint(self, state: Dict[str, Any], save_dir: str) -> str:
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is required. Install with: pip install tensorflow"
            ) from exc

        model = state.get("model")
        if model is None:
            raise ValueError("state must contain a 'model' key")
        if not isinstance(model, tf.keras.Model):
            raise TypeError(
                f"'model' must be a tf.keras.Model, got {type(model).__name__}"
            )

        version   = self._next_version(save_dir)
        final_dir = os.path.join(save_dir, f"v{version}")
        tmp_dir   = os.path.join(save_dir, f".tmp_v{version}")

        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        # 1. model
        model_path = os.path.join(tmp_dir, self.model_filename)
        model.save(model_path)
        logger.info("[KerasCheckpointManager] Model saved → %s", model_path)

        # 2. optimizer (optional)
        optimizer = state.get("optimizer")
        if optimizer is not None:
            try:
                opt_path = os.path.join(tmp_dir, self.OPTIMIZER_FILE)
                np.save(opt_path, np.array(optimizer.get_weights(), dtype=object), allow_pickle=True)
                logger.info("[KerasCheckpointManager] Optimizer saved → %s", opt_path)
            except Exception as e:
                logger.warning("[KerasCheckpointManager] Could not save optimizer weights: %s", e)

        # 3. metadata
        metadata = self._sanitize_metadata(state)
        metadata.update({
            "checkpoint_version": version,
            "model_filename":     self.model_filename,
            "model_checksum":     self._file_checksum(model_path),
        })
        with open(os.path.join(tmp_dir, self.METADATA_FILE), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        # 4. session info
        self.save_session_info(tmp_dir)

        # 5. atomic swap
        if os.path.exists(final_dir):
            shutil.rmtree(final_dir)
        os.rename(tmp_dir, final_dir)

        logger.info("[KerasCheckpointManager] Saved v%d → %s", version, final_dir)
        return final_dir

    def load_checkpoint(self, save_dir: str) -> Dict[str, Any]:
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is required. Install with: pip install tensorflow"
            ) from exc

        version = self._latest_valid_version(save_dir)
        if version == 0:
            raise FileNotFoundError("No valid checkpoints found in: " + save_dir)

        vdir = os.path.join(save_dir, f"v{version}")

        # metadata
        with open(os.path.join(vdir, self.METADATA_FILE), encoding="utf-8") as f:
            metadata = json.load(f)

        # model
        model_path = os.path.join(vdir, metadata.get("model_filename", self.model_filename))
        result = {
            **metadata,
            "model":              tf.keras.models.load_model(model_path),
            "checkpoint_version": version,
        }
        logger.info("[KerasCheckpointManager] Model loaded ← %s", model_path)

        # optimizer weights (optional)
        opt_path = os.path.join(vdir, self.OPTIMIZER_FILE)
        if os.path.exists(opt_path):
            result["optimizer_weights"] = list(np.load(opt_path, allow_pickle=True))
            logger.info("[KerasCheckpointManager] Optimizer loaded ← %s", opt_path)
        else:
            result["optimizer_weights"] = None

        # session info
        result["session_info"] = self.load_session_info(vdir)

        logger.info("[KerasCheckpointManager] Loaded v%d ← %s", version, vdir)
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_valid(self, save_dir: str, version: int) -> bool:
        vdir       = os.path.join(save_dir, f"v{version}")
        meta_path  = os.path.join(vdir, self.METADATA_FILE)
        model_path = os.path.join(vdir, self.model_filename)

        if not (os.path.exists(meta_path) and os.path.exists(model_path)):
            return False

        try:
            with open(meta_path, encoding="utf-8") as f:
                metadata = json.load(f)
            expected = metadata.get("model_checksum")
            return not expected or self._file_checksum(model_path) == expected
        except Exception:
            return False

    def _get_versions(self, base_dir: str) -> List[int]:
        p = Path(base_dir)
        if not p.exists():
            return []
        versions = []
        for d in p.iterdir():
            if d.is_dir():
                m = self.VERSION_RE.match(d.name)
                if m:
                    versions.append(int(m.group(1)))
        return sorted(versions)

    def _next_version(self, base_dir: str) -> int:
        versions = self._get_versions(base_dir)
        return 1 if not versions else versions[-1] + 1

    def _latest_valid_version(self, base_dir: str) -> int:
        for v in reversed(self._get_versions(base_dir)):
            if self._is_valid(base_dir, v):
                return v
        return 0

    def _sanitize_metadata(self, state: dict) -> dict:
        skip = {"model", "optimizer"}
        result = {}
        for k, v in state.items():
            if k in skip:
                continue
            try:
                json.dumps(v)
                result[k] = v
            except Exception:
                pass
        return result

    @staticmethod
    def _file_checksum(path: str) -> str:
        sha = hashlib.sha256()
        if os.path.isfile(path):
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    sha.update(chunk)
        elif os.path.isdir(path):
            for root, _, files in os.walk(path):
                for name in sorted(files):
                    with open(os.path.join(root, name), "rb") as f:
                        sha.update(f.read())
        return sha.hexdigest()