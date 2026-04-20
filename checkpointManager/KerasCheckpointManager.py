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
                optimizer_state.npz     <- optimizer config + weights packed together
                metadata.json           <- scalar/dict state fields + checksum
                session_info.json       <- written by CheckpointManager.save_session_info()

    Saves are atomic (temp dir → rename).  Each call creates a new version;
    load always picks the highest valid version (checksum-verified).

    Parameters
    ----------
    model_filename:
        Filename used for the saved Keras model inside each version dir.
    max_to_keep:
        Maximum number of versioned checkpoints to retain.  Oldest versions
        are deleted when the limit is exceeded.  ``None`` means keep all.
    """

    MODEL_FILE     = "model.keras"
    OPTIMIZER_FILE = "optimizer_state.npz"
    METADATA_FILE  = "metadata.json"
    VERSION_RE     = re.compile(r"^v(\d+)$")

    def __init__(self, max_to_keep:int=1, model_filename: str = "model.keras") -> None:
        super().__init__()
        self.max_to_keep = max_to_keep
        self.model_filename = model_filename

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_checkpoint(self, state: Dict[str, Any], save_dir: str) -> str:
        tf = self._import_tf()
        model = self._validate_model(state, tf)

        version = self._next_version(save_dir)
        tmp_dir, final_dir = self._version_dirs(save_dir, version)
        self._prepare_temp_dir(tmp_dir)

        model_path = self._save_model(model, tmp_dir)
        self._save_optimizer(state.get("optimizer"), tmp_dir, tf)
        self._save_metadata(state, model_path, version, tmp_dir)
        self.save_session_info(tmp_dir)
        self._atomic_swap(tmp_dir, final_dir)

        if self.max_to_keep is not None:
            self._prune(save_dir)

        logger.info("[KerasCheckpointManager] Saved v%d → %s", version, final_dir)
        return final_dir

    def load_checkpoint(self, save_dir: str) -> Dict[str, Any]:
        """
               Recover the most recent valid checkpoint from *save_dir*.

               Returns
               -------
               dict
                   Keys: all metadata fields, ``"model"`` (tf.keras.Model),
                   ``"optimizer_weights"`` (list of numpy arrays or ``None``),
                   ``"optimizer_config"`` (dict or ``None``),
                   ``"session_info"`` (dict or ``None``),
                   ``"checkpoint_version"`` (int).

               Notes
               -----
               To fully restore an optimizer, call::

                   optimizer = tf.keras.optimizers.deserialize(result["optimizer_config"])
                   optimizer.build(model.trainable_variables)
                   optimizer.set_weights(result["optimizer_weights"])
               """
        tf = self._import_tf()
        version = self._latest_valid_version(save_dir)
        if version ==0:
            raise FileNotFoundError("No valid checkpoints found in: " + save_dir)

        vdir = os.path.join(save_dir, f"v{version}")

        metadata = self._load_metadata(vdir)
        model = self._load_model(vdir, metadata, tf)
        opt_state = self._load_optimizer(vdir)
        session = self.load_session_info(vdir)
        result = {
            **metadata,
            "model": model,
            "optimizer": self._restore_optimizer(model, opt_state),  # ← add this
            "optimizer_weights": opt_state.get("weights"),
            "optimizer_config": opt_state.get("config"),
            "session_info": session,
            "checkpoint_version": version,
        }

        logger.info("[KerasCheckpointManager] Loaded v%d ← %s", version, vdir)
        return result

    def _restore_optimizer(self, model: Any, opt_state: Dict) -> Any:
        if not opt_state.get("config") or not opt_state.get("weights"):
            return None
        tf = self._import_tf()
        opt = tf.keras.optimizers.deserialize(opt_state["config"])
        opt.build(model.trainable_variables)
        opt.set_weights(opt_state["weights"])
        return opt

    # ------------------------------------------------------------------
    # Save helpers  (each does exactly one job)
    # ------------------------------------------------------------------
    @staticmethod
    def _import_tf() -> Any:
        try:
            import tensorflow as tf
            return tf
        except ImportError as axc:
            raise ImportError("TensorFlow is required. Install with: pip install tensorflow") from axc

    @staticmethod
    def _validate_model(state: Dict[str, Any], tf:Any) -> Any:
        model = state.get("model")
        if model is None:
            raise ValueError("state must contain a 'model' key")
        if not isinstance(model, tf.keras.Model):
            raise TypeError(f"'model' must be a tf.keras.Model, got {type(model).__name__}")
        return model



    def _save_model(self, model:Any, tmp_dir:str) -> str:
        model_path = os.path.join(tmp_dir, self.model_filename)
        model.save(model_path)
        logger.info("[KerasCheckpointManager] Model  saved -> %s", model_path)
        return model_path

    def _save_optimizer(self, optimizer: Any, tmp_dir: str, tf: Any) -> None:
        """
        Save optimizer config + weights together into a single .npz file.

        Why config + weights together?
        --------------------------------
        ``optimizer.get_weights()`` returns the *numerical* slot values
        (momentum, variance, …).  Without the config (learning-rate,
        beta_1, …) a loader cannot reconstruct the optimizer correctly.
        Storing both in one file keeps them in sync and makes the load
        side unambiguous.

        Edge-case: get_weights() returns [] when the optimizer has never
        been applied (no train step has run yet).  We still save the
        config so the loader can at least reconstruct the optimizer type
        and hyper-parameters.
        """
        if optimizer is None:
            return
        if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
            logger.warning("[KerasCheckpointManager] Optimizer not a tf.keras.optimizer - skipped")
            return
        try :
            config = optimizer.get_config()
            # Handle older TF versions without get_weights()
            if hasattr(optimizer, 'get_weights'):
                weights = optimizer.get_weights()
            else:
                # Fallback: try to get slot variables
                weights = []
                if hasattr(optimizer, 'variables'):
                    weights = optimizer.variables()
                logger.warning("[KerasCheckpointManager] Using fallback for optimizer weights")
            opt_path = os.path.join(tmp_dir, self.OPTIMIZER_FILE)
            np.savez(opt_path, config=np.array(json.dumps(config), dtype=object),
                weights=np.array(weights, dtype=object),)
            logger.info("[KerasCheckpointManager] Optimizer saved → %s  (weights=%d arrays)",opt_path, len(weights),)

        except Exception as e:
            logger.warning("[KerasCheckpointManager] Could not save optimizer weights: %s", e)


    # Load helpers  (each does exactly one job)

    def _load_model(self, vdir: str, metadata: Dict[str, Any], tf: Any) -> Any:
        filename = metadata.get("model_filename", self.model_filename)
        model_path = os.path.join(vdir, filename)
        model = tf.keras.models.load_model(model_path)
        logger.info("[KerasCheckpointManager] Model loaded ← %s", model_path)
        return model

    def _load_optimizer(self, vdir: str) -> Dict[str, Any]:
        """
                Returns ``{"config": dict, "weights": list}`` or ``{}`` if no
                optimizer file exists.

                The caller is responsible for calling::

                    opt = tf.keras.optimizers.deserialize(result["optimizer_config"])
                    opt.build(model.trainable_variables)
                    opt.set_weights(result["optimizer_weights"])

                We do *not* call set_weights here because this manager does not
                hold a reference to the live optimizer object.
                """
        opt_path = os.path.join(vdir, self.OPTIMIZER_FILE)
        if not os.path.exists(opt_path):
            return {}
        try:
            data = np.load(opt_path, allow_pickle=True)
            config = json.loads(str(data["config"]))
            weights = list(data["weights"])
            logger.info("[KerasCheckpointManager] Optimizer loaded ← %s  (weights=%d arrays)",opt_path, len(weights),)
            return {"config": config, "weights": weights}
        except Exception as e:
            logger.warning("[KerasCheckpointManager] Could not load optimizer weights: %s", e)
            return {}


# REPEATED Functions !!

    def _load_metadata(self, vdir: str) -> Dict[str, Any]:
        path = os.path.join(vdir, self.METADATA_FILE)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


    def _save_metadata(self, state: Dict[str, Any], model_path: str, version: int, tmp_dir: str,) -> None:
        metadata = self._sanitize_metadata(state)
        metadata.update({
            "checkpoint_version": version,
            "model_filename": self.model_filename,
            "model_checksum": self._file_checksum(model_path),
        })
        path = os.path.join(tmp_dir, self.METADATA_FILE)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.debug("[KerasCheckpointManager] Metadata written → %s", path)

    @staticmethod
    def _atomic_swap(tmp_dir: str, final_dir: str) -> None:
        if os.path.exists(final_dir):
            shutil.rmtree(final_dir)
        os.rename(tmp_dir, final_dir)

    @staticmethod
    def _prepare_temp_dir(temp_dir: str) -> None:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Versioning helpers
    # ------------------------------------------------------------------
    def _version_dirs(self, save_dir: str, version: int) -> tuple[str, str]:
        final_dir = os.path.join(save_dir, f"v{version}")
        tmp_dir = os.path.join(save_dir, f".tmp_v{version}")
        return tmp_dir, final_dir

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

    def _is_valid(self, save_dir: str, version: int) -> bool:
        vdir = os.path.join(save_dir, f"v{version}")
        meta_path = os.path.join(vdir, self.METADATA_FILE)
        model_path = os.path.join(vdir, self.model_filename)

        if not (os.path.exists(meta_path) and os.path.exists(model_path)):
            return False
        try:
            with open(meta_path, encoding="utf-8") as fh:
                metadata = json.load(fh)
            expected = metadata.get("model_checksum")
            return not expected or self._file_checksum(model_path) == expected
        except Exception:
            return False

    def _prune(self, base_dir: str) -> None:
        """Delete the oldest versions so that at most ``max_to_keep`` remain."""
        if self.max_to_keep is None:
            return
        versions = self._get_versions(base_dir)
        for old_version in versions[: -self.max_to_keep]:
            old_dir = os.path.join(base_dir, f"v{old_version}")
            try:
                shutil.rmtree(old_dir)
                logger.info(
                    "[KerasCheckpointManager] Pruned old checkpoint: %s", old_dir
                )
            except OSError:
                pass  # already removed — harmless

        # ------------------------------------------------------------------
        # Misc helpers
        # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_metadata(state: Dict[str, Any]) -> Dict[str, Any]:
        skip = {"model", "optimizer"}
        result: Dict[str, Any] = {}
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
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(65_536), b""):
                    sha.update(chunk)
        elif os.path.isdir(path):
            for root, _, files in os.walk(path):
                for name in sorted(files):
                    with open(os.path.join(root, name), "rb") as fh:
                        sha.update(fh.read())
        return sha.hexdigest()