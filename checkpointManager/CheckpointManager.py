from __future__ import annotations

import abc
import json
import logging
import os
import socket
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

SESSION_SCHEMA_VERSION = 1


class CheckpointManager(abc.ABC):

    SESSION_FILE = "session_info.json"

    @abc.abstractmethod
    def save_checkpoint(self, state: Dict[str, Any], save_dir: str) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def load_checkpoint(self, save_dir: str) -> Dict[str, Any]:
        raise NotImplementedError

    def save_session_info(self, save_dir: str, checkpoint_path: Optional[str] = None) -> None:
        os.makedirs(save_dir, exist_ok=True)

        session_data = {
            "version": SESSION_SCHEMA_VERSION,
            "last_save_time": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "checkpoint_manager": type(self).__name__,
            "checkpoint_path": checkpoint_path,
        }

        path = os.path.join(save_dir, self.SESSION_FILE)
        tmp_path = path + ".tmp"

        try:
            with open(tmp_path, "w", encoding="utf-8") as fh:
                json.dump(session_data, fh, indent=2)
            os.replace(tmp_path, path)  # atomic on POSIX and Windows

            logger.info("[CheckpointManager] Session info saved -> %s", path)

        except Exception as e:
            # Clean up the temp file if it was created
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            logger.exception("Failed to save session info")
            raise RuntimeError(f"Failed to save session info: {e}") from e

    def load_session_info(self, save_dir: str) -> Optional[Dict[str, Any]]:
        path = os.path.join(save_dir, self.SESSION_FILE)

        if not os.path.exists(path):
            logger.warning("[CheckpointManager] No session info found in %s", save_dir)
            return None

        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)

        except json.JSONDecodeError as e:
            logger.warning(
                "[CheckpointManager] Session info in %s is corrupt and will be ignored: %s",
                path, e,
            )
            return None

        except Exception as e:
            logger.exception("Failed to load session info")
            raise RuntimeError(f"Failed to load session info: {e}") from e

        # Schema validation
        if not isinstance(data, dict):
            logger.warning("[CheckpointManager] Session info has unexpected format in %s", path)
            return None

        file_version = data.get("version")
        if file_version is None:
            logger.warning(
                "[CheckpointManager] Session info in %s has no version field — "
                "likely written by an older build. Proceeding with caution.",
                path,
            )
        elif file_version != SESSION_SCHEMA_VERSION:
            logger.warning(
                "[CheckpointManager] Session info version mismatch in %s: "
                "expected %d, got %d. Some fields may be missing.",
                path, SESSION_SCHEMA_VERSION, file_version,
            )

        logger.info("[CheckpointManager] Session info loaded <- %s", path)
        return data
