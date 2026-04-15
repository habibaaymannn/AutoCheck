from __future__ import annotations

import logging
import os
from typing import Any, Dict

import numpy as np

from CheckpointManager import CheckpointManager

logger = logging.getLogger(__name__)


class KerasCheckpointManager(CheckpointManager):
    """
    Checkpoint manager for Keras / TensorFlow models.

    Inherits session info load/save from CheckpointManager and
    implements save_checkpoint / load_checkpoint using the native
    Keras SavedModel format (.keras file) together with a companion
    JSON file that carries all scalar metadata (epoch, step, …).

    Directory layout produced by save_checkpoint():
        <save_dir>/
            model.keras          ← full Keras model (weights + architecture + compile state)
            optimizer_state.npy  ← optimizer weights as a numpy .npy file (optional)
            metadata.json        ← every non-model key from the state dict
            session_info.json    ← written by CheckpointManager.save_session_info()

    Parameters
    ----------
    model_filename : str
        Name of the Keras model file inside save_dir.
        Defaults to "model.keras".
    """

    MODEL_FILE = "model.keras"
    OPTIMIZER_FILE = "optimizer_state.npy"
    METADATA_FILE = "metadata.json"

    def __init__(self, model_filename: str = "model.keras") -> None:
        self.model_filename = model_filename

    def save_checkpoint(self, state: Dict[str, Any], save_dir: str) -> str:
        """
        Persist a full training snapshot to *save_dir*.

        Expected keys in *state*
        ------------------------
        model          : tf.keras.Model  — the live Keras model object
        optimizer      : (optional) tf.keras.optimizers.Optimizer
        epoch          : int
        global_step    : int  (optional)
        batch_idx      : int  (optional)
        …any extra scalar / dict fields are stored in metadata.json

        Returns
        -------
        str  — absolute path to the directory that was written.
        """
        try:
            import tensorflow as tf  # local import — TF may not be installed
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is required for KerasCheckpointManager. "
                "Install it with: pip install tensorflow"
            ) from exc

        os.makedirs(save_dir, exist_ok=True)

        # ── 1. Save the Keras model ───────────────────────────────────
        model: tf.keras.Model = state.get("model")
        if model is None:
            raise KeyError("'model' key is missing from state — cannot save Keras checkpoint.")
        if not isinstance(model, tf.keras.Model):
            raise TypeError(
                f"'model' must be a tf.keras.Model instance, got {type(model).__name__}."
            )

        model_path = os.path.join(save_dir, self.model_filename)
        model.save(model_path)
        logger.info("[KerasCKPT] Model saved → %s", model_path)

        # ── 2. Save optimizer weights (optional) ─────────────────────
        optimizer = state.get("optimizer")
        if optimizer is not None:
            try:
                opt_weights = optimizer.get_weights()
                opt_path = os.path.join(save_dir, self.OPTIMIZER_FILE)
                np.save(opt_path, np.array(opt_weights, dtype=object), allow_pickle=True)
                logger.info("[KerasCKPT] Optimizer weights saved → %s", opt_path)
            except Exception as e:
                logger.warning("[KerasCKPT] Could not save optimizer weights: %s", e)

        # ── 3. Save scalar / dict metadata ───────────────────────────
        import json

        skip_keys = {"model", "optimizer"}
        metadata: Dict[str, Any] = {
            k: v for k, v in state.items()
            if k not in skip_keys and self._is_json_serialisable(v)
        }

        meta_path = os.path.join(save_dir, self.METADATA_FILE)
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)
        logger.info("[KerasCKPT] Metadata saved → %s", meta_path)

        # ── 4. Session info ───────────────────────────────────────────
        self.save_session_info(save_dir)

        logger.info("[KerasCKPT] Checkpoint complete → %s", save_dir)
        return os.path.abspath(save_dir)

    def load_checkpoint(self, save_dir: str) -> Dict[str, Any]:
        """
        Load a previously saved checkpoint from *save_dir*.

        Returns
        -------
        dict with keys:
            model          : tf.keras.Model   (fully restored)
            optimizer_weights : list | None   (raw weight arrays — caller must
                                               call optimizer.set_weights() after
                                               the first training step)
            epoch, global_step, batch_idx, … (all scalars from metadata.json)
            session_info   : dict | None      (from session_info.json)
        """
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ImportError(
                "TensorFlow is required for KerasCheckpointManager. "
                "Install it with: pip install tensorflow"
            ) from exc

        import json

        if not os.path.isdir(save_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {save_dir}")

        state: Dict[str, Any] = {}

        # ── 1. Load the Keras model ───────────────────────────────────
        model_path = os.path.join(save_dir, self.model_filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        state["model"] = tf.keras.models.load_model(model_path)
        logger.info("[KerasCKPT] Model loaded ← %s", model_path)

        # ── 2. Load optimizer weights (if present) ────────────────────
        opt_path = os.path.join(save_dir, self.OPTIMIZER_FILE)
        if os.path.exists(opt_path):
            opt_weights = list(np.load(opt_path, allow_pickle=True))
            state["optimizer_weights"] = opt_weights
            logger.info("[KerasCKPT] Optimizer weights loaded ← %s", opt_path)
        else:
            state["optimizer_weights"] = None
            logger.warning("[KerasCKPT] No optimizer weights file found in %s", save_dir)

        # ── 3. Load metadata ──────────────────────────────────────────
        meta_path = os.path.join(save_dir, self.METADATA_FILE)
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as fh:
                metadata = json.load(fh)
            state.update(metadata)
            logger.info("[KerasCKPT] Metadata loaded ← %s", meta_path)
        else:
            logger.warning("[KerasCKPT] No metadata.json found in %s", save_dir)

        # ── 4. Session info ───────────────────────────────────────────
        state["session_info"] = self.load_session_info(save_dir)

        logger.info("[KerasCKPT] Checkpoint loaded ← %s", save_dir)
        return state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_json_serialisable(value: Any) -> bool:
        """Return True if *value* can be written directly to JSON."""
        import json
        try:
            json.dumps(value)
            return True
        except (TypeError, ValueError):
            return False
