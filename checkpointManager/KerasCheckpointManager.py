from __future__ import annotations
import tensorflow as tf
import hashlib
import json
import logging
import os
import shutil
import tempfile
import re
from pathlib import Path
from typing import Any, Dict, List
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
    OPT_DIR = "optimizer"
    METADATA_FILE = "metadata.json"

    def __init__(self, model_filename: str = "model.keras") -> None:
        self.model_filename = model_filename

    def save_checkpoint(self, state: Dict[str, Any], save_dir: str) -> str:
        model = state.get("model")
        if model is None:
            raise ValueError("Missing model")
        version = self.next_version(save_dir)

        # final target for this version
        final_version_dir = os.path.join(save_dir, f"v{version}")

        # temp directory INSIDE save_dir (safe + atomic per version)
        temp_version_dir = os.path.join(save_dir, f".tmp_v{version}")

        # clean temp if exists
        if os.path.exists(temp_version_dir):
            shutil.rmtree(temp_version_dir)

        os.makedirs(temp_version_dir, exist_ok=True)
        model_path = os.path.join(temp_version_dir, self.MODEL_FILE)
        model.save(model_path)

        # 2. optimizer
        if state.get("optimizer") is not None:
            opt_dir = os.path.join(temp_version_dir, self.OPT_DIR)
            os.makedirs(opt_dir, exist_ok=True)

            ckpt = tf.train.Checkpoint(optimizer=state["optimizer"])
            ckpt.write(os.path.join(opt_dir, "ckpt"))

        # 3. metadata
        metadata = self.sanitize_metadata(state)
        metadata.update({
            "checkpoint_version": version,
            "model_filename": self.MODEL_FILE,
            "model_checksum": self.file_checksum(model_path)
        })

        with open(os.path.join(temp_version_dir, self.METADATA_FILE), "w") as f:
            json.dump(metadata, f, indent=2)

        # 4. swap atomically
        final_version_dir = os.path.join(save_dir, f"v{version}")

        # 4. atomic move into final version folder
        if os.path.exists(final_version_dir):
            shutil.rmtree(final_version_dir)

        os.rename(temp_version_dir, final_version_dir)

        return final_version_dir

    def load_checkpoint(self, save_dir: str) -> Dict[str, Any]:
        latest = self.latest_version(save_dir)
        if latest == 0:
            raise FileNotFoundError("No checkpoints found")

        version = self.latest_valid_version(
            save_dir,
            validator=lambda v: self._valid(save_dir, v)
        )

        if version == 0:
            raise RuntimeError("No valid checkpoints")

        vdir = os.path.join(save_dir, f"v{version}")

        # metadata
        with open(os.path.join(vdir, self.METADATA_FILE)) as f:
            metadata = json.load(f)

        # model
        model_path = os.path.join(vdir, metadata.get("model_filename", self.MODEL_FILE))
        model = tf.keras.models.load_model(model_path)

        result = dict(metadata)
        result["model"] = model
        result["checkpoint_version"] = version

        # optimizer path only (no closure!)
        opt_path = os.path.join(vdir, self.OPT_DIR, "ckpt")
        result["optimizer_ckpt_path"] = opt_path if os.path.exists(opt_path) else None

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _valid(self, save_dir: str, version: int) -> bool:
        vdir = os.path.join(save_dir, f"v{version}")
        return (
                os.path.exists(os.path.join(vdir, self.METADATA_FILE)) and
                os.path.exists(os.path.join(vdir, self.MODEL_FILE))
        )

    def is_json_serializable(self, v: Any) -> bool:
        try:
            json.dumps(v)
            return True
        except Exception:
            return False

    def sanitize_metadata(self, state: dict) -> dict:
        return {
            k: v
            for k, v in state.items()
            if k not in {"model", "optimizer"} and self.is_json_serializable(v)
        }


    def file_checksum(self, path: str) -> str:
        sha = hashlib.sha256()

        if os.path.isfile(path):
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    sha.update(chunk)

        elif os.path.isdir(path):
            for root, _, files in os.walk(path):
                for f in sorted(files):
                    with open(os.path.join(root, f), "rb") as file:
                        sha.update(file.read())

        return sha.hexdigest()

    VERSION_PATTERN = re.compile(r"v(\d+)")

    def get_versions(self, base_dir: str) -> List[int]:
        path = Path(base_dir)
        if not path.exists():
            return []

        versions = []
        for item in path.iterdir():
            if item.is_dir():
                match = self.VERSION_PATTERN.match(item.name)
                if match:
                    versions.append(int(match.group(1)))

        return sorted(versions)

    def next_version(self, base_dir: str) -> int:
        versions = self.get_versions(base_dir)
        return 1 if not versions else versions[-1] + 1

    def latest_version(self, base_dir: str) -> int:
        versions = self.get_versions(base_dir)
        return versions[-1] if versions else 0

    def latest_valid_version(self, base_dir: str, validator) -> int:
        for v in reversed(self.get_versions(base_dir)):
            if validator(v):
                return v
        return 0


    def create_temp_dir(self, base_dir: str):
        parent = os.path.dirname(os.path.abspath(base_dir)) or "."
        os.makedirs(parent, exist_ok=True)
        return tempfile.TemporaryDirectory(dir=parent, prefix=".tmp_ckpt_")

    def atomic_replace(self, tmp_dir: str, target_dir: str):
        tmp_path = Path(tmp_dir)
        target_path = Path(target_dir)

        backup = None

        if target_path.exists():
            backup = target_path.with_suffix(".bak")
            if backup.exists():
                shutil.rmtree(backup)
            os.rename(target_path, backup)

        os.rename(tmp_path, target_path)

        if backup and backup.exists():
            shutil.rmtree(backup)