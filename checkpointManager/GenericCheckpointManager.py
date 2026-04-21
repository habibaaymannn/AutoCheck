from __future__ import annotations

import logging
import os
import re
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from checkpointManager.CheckpointManager import CheckpointManager
from checkpointManager.serializers import JobLibSerializer, Serializer

logger = logging.getLogger(__name__)

CHECKPOINT_VERSION = 1


class GenericCheckpointManager(CheckpointManager):
    """
    Framework-agnostic checkpoint manager using pluggable serializers.
    """

    def __init__(
        self,
        checkpoint_dir: str,
        serializer: Optional[Serializer] = None,
        max_to_keep: Optional[int] = None,
    ) -> None:

        self._dir = Path(checkpoint_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

        self._serializer: Serializer = serializer or JobLibSerializer()
        self.max_to_keep = max_to_keep

        ext = re.escape(self._serializer.extension)
        self._ckpt_re = re.compile(rf"^checkpoint_(.+){ext}$")

        logger.info(
            "[GenericCheckpointManager] Initialized | dir=%s | serializer=%s",
            self._dir,
            self._serializer,
        )

    # =========================
    # Public API
    # =========================

    def save_checkpoint(self, state: Dict[str, Any], save_dir: str) -> str:

        directory = Path(save_dir)
        directory.mkdir(parents=True, exist_ok=True)

        # Flexible naming
        if "step" in state:
            name = f"{int(state['step']):08d}"
        else:
            name = str(int(time.time()))

        path = directory / f"checkpoint_{name}{self._serializer.extension}"

        payload = {
            "version": CHECKPOINT_VERSION,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "state": state,
        }

        self._atomic_dump(payload, path)

        logger.info("[CheckpointManager] Saved -> %s", path)

        self.save_session_info(save_dir, checkpoint_path=str(path))

        if self.max_to_keep:
            self._prune(directory)

        return str(path)

    def load_checkpoint(self, save_dir: str) -> Dict[str, Any]:

        directory = Path(save_dir)

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        latest = self._get_latest_checkpoint(directory)

        if not latest:
            raise RuntimeError("No checkpoints found")

        try:
            payload = self._serializer.load(latest)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}") from e

        if "state" not in payload:
            raise RuntimeError("Corrupted checkpoint: missing 'state'")

        logger.info("[CheckpointManager] Loaded <- %s", latest)

        return payload["state"]

    # =========================
    # Helpers
    # =========================

    def _atomic_dump(self, payload: Dict[str, Any], path: Path) -> None:

        tmp_fd, tmp_name = tempfile.mkstemp(
            dir=path.parent,
            suffix=".tmp",
        )

        try:
            os.close(tmp_fd)
            self._serializer.dump(payload, Path(tmp_name))
            os.replace(tmp_name, path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise

    def _get_latest_checkpoint(self, directory: Path) -> Optional[Path]:
        files = self._list_checkpoints(directory)
        return files[-1] if files else None

    def _list_checkpoints(self, directory: Path) -> List[Path]:

        if not directory.exists():
            return []

        found: List[tuple[str, Path]] = []

        for p in directory.iterdir():
            m = self._ckpt_re.match(p.name)
            if m:
                found.append((m.group(1), p))

        found.sort(key=lambda t: t[0])
        return [p for _, p in found]

    def _prune(self, directory: Path) -> None:

        files = self._list_checkpoints(directory)

        if len(files) <= self.max_to_keep:
            return

        for old in files[: -self.max_to_keep]:
            try:
                old.unlink()
                logger.debug("Pruned: %s", old)
            except OSError:
                pass