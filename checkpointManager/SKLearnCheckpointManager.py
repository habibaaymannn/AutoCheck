from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib

from .CheckpointManager import CheckpointManager

logger = logging.getLogger(__name__)


class SKLearnCheckpointManager(CheckpointManager):
    MODEL_FILE    = "model.joblib"
    METADATA_FILE = "metadata.json"
    VERSION_RE    = re.compile(r"^v(\d+)$")

    def __init__(self, checkpoint_dir: str, max_to_keep: Optional[int] = None):
        self._dir = Path(checkpoint_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_checkpoint(self, state: Dict[str, Any], save_dir: str) -> str:
        """
                Persist the estimator and all serializable metadata to a new
                versioned directory under *save_dir*.

                Parameters
                ----------
                state:
                    Must contain ``"model"`` (a fitted sklearn estimator or pipeline).
                    Any additional JSON-serializable keys are saved as metadata.

                Returns
                -------
                str
                    Absolute path of the newly created version directory.
        """
        model = self._validate_model(state)

        version = self._next_version(save_dir)
        tmp_dir, final_dir = self._version_dirs(save_dir, version)

        self._prepare_temp_dir(tmp_dir)
        model_path = self._save_model(model, tmp_dir)
        self._save_metadata(state, model_path, version, tmp_dir)
        self.save_session_info(save_dir, checkpoint_path=final_dir)
        self._atomic_swap(tmp_dir, final_dir)

        if self.max_to_keep is not None:
            self._prune(save_dir)

        logger.info("[SKLearnCheckpointManager] Saved v%d → %s", version, final_dir)
        return final_dir

    def load_checkpoint(self, save_dir: str) -> Dict[str, Any]:
        """
        Recover the most recent valid checkpoint from *save_dir*.

        Returns
        -------
        dict
            Keys: all metadata fields, ``"model"`` (fitted estimator),
            ``"session_info"`` (dict or ``None``),
            ``"checkpoint_version"`` (int).

        Raises
        ------
        FileNotFoundError
            If no valid checkpoints are found in *save_dir*.
        """
        version = self._latest_valid_version(save_dir)
        if version == 0:
            raise FileNotFoundError("No valid checkpoints found in: " + save_dir)

        vdir = os.path.join(save_dir, f"v{version}")

        metadata = self._load_metadata(vdir)
        model    = self._load_model(vdir, metadata)
        session  =self.load_session_info(save_dir)
        result = {
            **metadata,
            "model":              model,
            "session_info":       session,
            "checkpoint_version": version,
        }

        logger.info("[SKLearnCheckpointManager] Loaded v%d ← %s", version, vdir)
        return result
    # ------------------------------------------------------------------
    # Save helpers  (each does exactly one job)
    # ------------------------------------------------------------------
    def _save_model(self, model: Any, tmp_dir: str) -> str:
        model_path = os.path.join(tmp_dir, self.MODEL_FILE)
        joblib.dump(model, model_path, compress=3)
        logger.info("[SKLearnCheckpointManager] Model saved → %s", model_path)
        return model_path

    # ------------------------------------------------------------------
    # Load helpers  (each does exactly one job)
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_model(state: Dict[str, Any]) -> Any:
        model = state.get("model")
        if model is None:
            raise ValueError("state must contain a 'model' key")
        return model

    def _load_model(self, vdir: str, metadata: Dict[str, Any]) -> Any:
        filename = metadata.get("model_filename", self.MODEL_FILE)
        model_path = os.path.join(vdir, filename)
        model = joblib.load(model_path)
        logger.info("[SKLearnCheckpointManager] Model loaded ← %s", model_path)
        return model


        # REPEATED Functions !!
    def _load_metadata(self, vdir: str) -> Dict[str, Any]:
        path = os.path.join(vdir, self.METADATA_FILE)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_metadata(self, state: Dict[str, Any], model_path: str, version: int, tmp_dir: str, ) -> None:
        metadata = self._sanitize_metadata(state)
        metadata.update({
            "checkpoint_version": version,
            "model_filename": self.MODEL_FILE,
            "model_checksum": self._file_checksum(model_path),
        })
        path = os.path.join(tmp_dir, self.METADATA_FILE)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.debug("[SKLearnCheckpointManager] Metadata written → %s", path)

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
        model_path = os.path.join(vdir, self.MODEL_FILE)

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
                    "[SKLearnCheckpointManager] Pruned old checkpoint: %s", old_dir
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