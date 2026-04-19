from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from typing import Any, Dict

import joblib

from .CheckpointManager import CheckpointManager

logger = logging.getLogger(__name__)


# No module-level CHECKPOINT_VERSION — versions are derived from the filesystem.


class SKLearnCheckpointManager(CheckpointManager):
    """
    Checkpoint manager for scikit-learn estimators.

    Inherits session info load/save from CheckpointManager and
    implements save_checkpoint / load_checkpoint using joblib for the
    estimator (the standard sklearn recommendation) and JSON for all
    scalar / dict metadata.

    Directory layout produced by save_checkpoint():
        <save_dir>/
            v{checkpoint_version}/
                estimator.joblib     <- serialised sklearn estimator
                metadata.json        <- scalar training metadata + version + checksum
                session_info.json    <- written by CheckpointManager.save_session_info()

    Each call to save_checkpoint() creates a new v{n+1}/ subdirectory,
    preserving all previous versions.  load_checkpoint() always loads the
    highest-numbered version found on disk.

    Writes are atomic: everything goes to a temp dir first, then the
    temp dir is swapped into place so a crash mid-save never leaves a
    partial checkpoint.

    Parameters
    ----------
    estimator_filename : str
        Name of the joblib file inside the versioned subdirectory.
        Defaults to "estimator.joblib".
    compress : int
        joblib compression level (0 = none, 1-9 = zlib). Defaults to 3.
    """

    ESTIMATOR_FILE = "estimator.joblib"
    METADATA_FILE = "metadata.json"

    def __init__(
            self,
            estimator_filename: str = "estimator.joblib",
            compress: int = 3,
    ) -> None:
        self.estimator_filename = estimator_filename
        self.compress = compress

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_checkpoint(self, state: Dict[str, Any], save_dir: str) -> str:
        """
        Persist a full sklearn training snapshot to *save_dir* atomically.

        Each call appends a new versioned subdirectory (v1/, v2/, …).
        The version number is derived from what already exists on disk:
            - save_dir is empty or missing  -> writes to v1/
            - highest existing version is n -> writes to v{n+1}/

        Expected keys in *state*
        ------------------------
        estimator       : sklearn estimator  — any fitted sklearn-compatible object
        iteration       : int  (optional) — e.g. current CV fold, warm_start epoch
        last_completed_unit : int (optional)
        params          : dict (optional) — hyper-parameters used
        metrics         : dict (optional) — validation scores, etc.
        ...any extra JSON-serialisable fields are stored in metadata.json

        Returns
        -------
        str  — absolute path to the versioned subdirectory that was written.
        """
        # ── Guard: save_dir must not already exist as a file ─────────
        if os.path.isfile(save_dir):
            raise ValueError(
                f"save_dir '{save_dir}' already exists as a file. "
                "Provide a directory path instead."
            )

        # ── Validate estimator ────────────────────────────────────────
        estimator = state.get("estimator")
        if estimator is None:
            raise KeyError(
                "'estimator' key is missing from state — cannot save SKLearn checkpoint."
            )
        if not (hasattr(estimator, "get_params") and callable(estimator.get_params)):
            raise TypeError(
                f"'estimator' does not look like a sklearn estimator "
                f"(no get_params()). Got: {type(estimator).__name__}."
            )

        # ── Determine next version from what already exists on disk ───
        next_version = self._resolve_next_version(save_dir)
        logger.info("[SKLearnCKPT] Saving as version v%d -> %s", next_version, save_dir)

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

            # 1. Save the estimator with joblib
            estimator_path = os.path.join(version_subdir, self.estimator_filename)
            joblib.dump(estimator, estimator_path, compress=self.compress)
            logger.info("[SKLearnCKPT] Estimator saved -> %s", estimator_path)

            # 2. Build and save metadata (with version + estimator checksum)
            skip_keys = {"estimator"}
            metadata: Dict[str, Any] = {}

            for k, v in state.items():
                if k in skip_keys:
                    continue
                if self._is_json_serialisable(v):
                    metadata[k] = v
                else:
                    logger.warning(
                        "[SKLearnCKPT] Skipping non-serialisable key '%s' (type=%s)",
                        k, type(v).__name__,
                    )

            # Always persist class name, hyper-params, version, and checksum
            metadata.setdefault("estimator_class", type(estimator).__name__)
            try:
                metadata.setdefault("estimator_params", estimator.get_params(deep=True))
            except Exception as e:
                logger.warning("[SKLearnCKPT] Could not read estimator params: %s", e)

            metadata["checkpoint_version"] = next_version
            metadata["estimator_filename"] = self.estimator_filename
            metadata["estimator_checksum"] = self._file_checksum(estimator_path)

            meta_path = os.path.join(version_subdir, self.METADATA_FILE)
            with open(meta_path, "w", encoding="utf-8") as fh:
                json.dump(metadata, fh, indent=2)
            logger.info("[SKLearnCKPT] Metadata saved -> %s", meta_path)

            # 3. Session info
            self.save_session_info(version_subdir)

            # 4. Atomic swap: replace save_dir with the fully prepared tmp dir
            if os.path.isdir(save_dir):
                shutil.rmtree(save_dir)
            shutil.copytree(tmp_dir, save_dir)

        logger.info("[SKLearnCKPT] Checkpoint complete -> %s", save_dir)
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
            estimator        : sklearn estimator  (fully restored, ready to predict)
            estimator_class  : str                (class name, for logging / assertions)
            estimator_params : dict               (hyper-parameters used at save time)
            checkpoint_version : int              (the version that was loaded)
            iteration, last_completed_unit, metrics, ... (all scalars from metadata.json)
            session_info     : dict | None        (from session_info.json)
        """
        if not os.path.isdir(save_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {save_dir}")

        # Auto-detect the latest versioned subdirectory (e.g. save_dir/v3/)
        latest_version = self._resolve_latest_version(save_dir)
        version_subdir = os.path.join(save_dir, f"v{latest_version}")
        logger.info("[SKLearnCKPT] Loading latest checkpoint v%d <- %s", latest_version, save_dir)

        state: Dict[str, Any] = {}

        # 1. Load metadata first (so we can read estimator_filename + checksum)
        meta_path = os.path.join(version_subdir, self.METADATA_FILE)
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"metadata.json not found in '{version_subdir}'. "
                "The checkpoint may be corrupt."
            )
        with open(meta_path, "r", encoding="utf-8") as fh:
            metadata = json.load(fh)

        state.update(metadata)
        logger.info("[SKLearnCKPT] Metadata loaded <- %s", meta_path)

        # 2. Load and verify the estimator
        estimator_filename = metadata.get("estimator_filename", self.estimator_filename)
        estimator_path = os.path.join(version_subdir, estimator_filename)
        if not os.path.exists(estimator_path):
            raise FileNotFoundError(f"Estimator file not found: {estimator_path}")

        saved_checksum = metadata.get("estimator_checksum")
        if saved_checksum is not None:
            actual_checksum = self._file_checksum(estimator_path)
            if actual_checksum != saved_checksum:
                raise ValueError(
                    f"Estimator file checksum mismatch for '{estimator_path}'. "
                    "The file may be corrupt."
                )

        state["estimator"] = joblib.load(estimator_path)
        logger.info("[SKLearnCKPT] Estimator loaded <- %s", estimator_path)

        # 3. Session info
        state["session_info"] = self.load_session_info(version_subdir)

        logger.info("[SKLearnCKPT] Checkpoint loaded <- %s", save_dir)
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
