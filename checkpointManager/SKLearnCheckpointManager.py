from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

import joblib

from CheckpointManager import CheckpointManager

logger = logging.getLogger(__name__)


class SKLearnCheckpointManager(CheckpointManager):
    """
    Checkpoint manager for scikit-learn estimators.

    Inherits session info load/save from CheckpointManager and
    implements save_checkpoint / load_checkpoint using joblib for the
    estimator (the standard sklearn recommendation) and JSON for all
    scalar / dict metadata.

    Directory layout produced by save_checkpoint():
        <save_dir>/
            estimator.joblib     ← serialised sklearn estimator
            metadata.json        ← scalar training metadata (iteration, params, …)
            session_info.json    ← written by CheckpointManager.save_session_info()

    Parameters
    ----------
    estimator_filename : str
        Name of the joblib file inside save_dir.
        Defaults to "estimator.joblib".
    compress : int
        joblib compression level (0 = none, 1–9 = zlib). Defaults to 3.
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
        Persist a full sklearn training snapshot to *save_dir*.

        Expected keys in *state*
        ------------------------
        estimator       : sklearn estimator  — any fitted sklearn-compatible object
        iteration       : int  (optional) — e.g. current CV fold, warm_start epoch
        last_completed_unit : int (optional)
        params          : dict (optional) — hyper-parameters used
        metrics         : dict (optional) — validation scores, etc.
        …any extra JSON-serialisable fields are stored in metadata.json

        Returns
        -------
        str  — absolute path to the directory that was written.
        """
        os.makedirs(save_dir, exist_ok=True)

        # ── 1. Save the estimator with joblib ─────────────────────────
        estimator = state.get("estimator")
        if estimator is None:
            raise KeyError(
                "'estimator' key is missing from state — cannot save SKLearn checkpoint."
            )

        # Duck-type check: sklearn estimators expose get_params()
        if not (hasattr(estimator, "get_params") and callable(estimator.get_params)):
            raise TypeError(
                f"'estimator' does not look like a sklearn estimator "
                f"(no get_params()). Got: {type(estimator).__name__}."
            )

        estimator_path = os.path.join(save_dir, self.estimator_filename)
        joblib.dump(estimator, estimator_path, compress=self.compress)
        logger.info("[SKLearnCKPT] Estimator saved → %s", estimator_path)

        # ── 2. Save scalar / dict metadata ───────────────────────────
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

        # Always persist estimator class name and its hyper-parameters
        metadata.setdefault("estimator_class", type(estimator).__name__)
        try:
            metadata.setdefault("estimator_params", estimator.get_params(deep=True))
        except Exception as e:
            logger.warning("[SKLearnCKPT] Could not read estimator params: %s", e)

        meta_path = os.path.join(save_dir, self.METADATA_FILE)
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)
        logger.info("[SKLearnCKPT] Metadata saved → %s", meta_path)

        # ── 3. Session info ───────────────────────────────────────────
        self.save_session_info(save_dir)

        logger.info("[SKLearnCKPT] Checkpoint complete → %s", save_dir)
        return os.path.abspath(save_dir)

    def load_checkpoint(self, save_dir: str) -> Dict[str, Any]:
        """
        Load a previously saved checkpoint from *save_dir*.

        Returns
        -------
        dict with keys:
            estimator      : sklearn estimator  (fully restored, ready to predict)
            estimator_class: str                (class name, for logging / assertions)
            estimator_params: dict              (hyper-parameters used at save time)
            iteration, last_completed_unit, metrics, … (all scalars from metadata.json)
            session_info   : dict | None        (from session_info.json)
        """
        if not os.path.isdir(save_dir):
            raise FileNotFoundError(f"Checkpoint directory not found: {save_dir}")

        state: Dict[str, Any] = {}

        # ── 1. Load the estimator ─────────────────────────────────────
        estimator_path = os.path.join(save_dir, self.estimator_filename)
        if not os.path.exists(estimator_path):
            raise FileNotFoundError(f"Estimator file not found: {estimator_path}")

        state["estimator"] = joblib.load(estimator_path)
        logger.info("[SKLearnCKPT] Estimator loaded ← %s", estimator_path)

        # ── 2. Load metadata ──────────────────────────────────────────
        meta_path = os.path.join(save_dir, self.METADATA_FILE)
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as fh:
                metadata = json.load(fh)
            state.update(metadata)
            logger.info("[SKLearnCKPT] Metadata loaded ← %s", meta_path)
        else:
            logger.warning("[SKLearnCKPT] No metadata.json found in %s", save_dir)

        # ── 3. Session info ───────────────────────────────────────────
        state["session_info"] = self.load_session_info(save_dir)

        logger.info("[SKLearnCKPT] Checkpoint loaded ← %s", save_dir)
        return state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_json_serialisable(value: Any) -> bool:
        """Return True if *value* can be written directly to JSON."""
        try:
            json.dumps(value)
            return True
        except (TypeError, ValueError):
            return False
