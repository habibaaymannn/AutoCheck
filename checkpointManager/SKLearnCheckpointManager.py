from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List

import joblib

from .CheckpointManager import CheckpointManager

logger = logging.getLogger(__name__)


class SKLearnCheckpointManager(CheckpointManager):
    """
    Checkpoint manager for scikit-learn estimators / pipelines.

    Designed for long-running jobs (GridSearchCV, incremental learning,
    cross-validation, etc.) — saves everything needed to resume.

    Directory layout:
        <save_dir>/
            v{n}/
                model.joblib         <- fitted sklearn estimator / pipeline
                metadata.json        <- epoch, step, cv_results, params, checksum, ...
                session_info.json    <- written by CheckpointManager.save_session_info()

    Saves are atomic (temp dir → rename). Each call creates a new version;
    load always picks the highest valid version (checksum-verified).

    State dict keys
    ---------------
    Required:
        model       : fitted sklearn estimator or pipeline

    Optional (all serializable values are saved automatically):
        step            : int   – current iteration / fold index
        epoch           : int   – epoch counter
        cv_results      : dict  – partial CV results so far
        best_params     : dict  – best params found so far
        best_score      : float – best score so far
        n_samples_seen  : int   – for incremental / partial_fit jobs
        classes         : list  – for incremental classifiers
        ... any other JSON-serializable key is saved as-is
    """

    MODEL_FILE    = "model.joblib"
    METADATA_FILE = "metadata.json"
    VERSION_RE    = re.compile(r"^v(\d+)$")

    def __init__(self, model_filename: str = "model.joblib") -> None:
        super().__init__()
        self.model_filename = model_filename

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_checkpoint(self, state: Dict[str, Any], save_dir: str) -> str:
        model = state.get("model")
        if model is None:
            raise ValueError("state must contain a 'model' key")

        version   = self._next_version(save_dir)
        final_dir = os.path.join(save_dir, f"v{version}")
        tmp_dir   = os.path.join(save_dir, f".tmp_v{version}")

        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)

        # 1. model
        model_path = os.path.join(tmp_dir, self.model_filename)
        joblib.dump(model, model_path, compress=3)

        # 2. metadata  (everything serializable except model)
        metadata = self._sanitize_metadata(state)
        metadata.update({
            "checkpoint_version": version,
            "model_filename":     self.model_filename,
            "model_checksum":     self._file_checksum(model_path),
        })
        with open(os.path.join(tmp_dir, self.METADATA_FILE), "w") as f:
            json.dump(metadata, f, indent=2)

        # 3. session info
        self.save_session_info(tmp_dir)

        # 4. atomic swap
        if os.path.exists(final_dir):
            shutil.rmtree(final_dir)
        os.rename(tmp_dir, final_dir)

        logger.info("[SKLearnCheckpointManager] Saved v%d → %s", version, final_dir)
        return final_dir

    def load_checkpoint(self, save_dir: str) -> Dict[str, Any]:
        version = self._latest_valid_version(save_dir)
        if version == 0:
            raise FileNotFoundError("No valid checkpoints found in: " + save_dir)

        vdir = os.path.join(save_dir, f"v{version}")

        with open(os.path.join(vdir, self.METADATA_FILE)) as f:
            metadata = json.load(f)

        model_path = os.path.join(vdir, metadata.get("model_filename", self.model_filename))

        result = {
            **metadata,
            "model":              joblib.load(model_path),
            "checkpoint_version": version,
            "session_info":       self.load_session_info(vdir),
        }


        logger.info("[SKLearnCheckpointManager] Loaded v%d ← %s", version, vdir)
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
            with open(meta_path) as f:
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
        skip = {"model"}
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
