from __future__ import annotations

import abc
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class CheckpointManager(abc.ABC):

    SESSION_FILE = "session_info.json"

    @abc.abstractmethod
    def save_checkpoint(self, state: Dict[str, Any], save_dir: str) -> str:
        raise NotImplementedError

    @abc.abstractmethod
    def load_checkpoint(self, save_dir: str) -> Dict[str, Any]:
        raise NotImplementedError

    
    def save_session_info(self, save_dir: str) -> None:
      
        os.makedirs(save_dir, exist_ok=True)

        session_data = {
            "last_save_time": datetime.now().isoformat(timespec="minutes"),
        }

        path = os.path.join(save_dir, self.SESSION_FILE)

        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(session_data, fh, indent=2)

            logger.info("[CheckpointManager] Session info saved → %s", path)

        except Exception as e:
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

            logger.info("[CheckpointManager] Session info loaded ← %s", path)
            return data

        except Exception as e:
            logger.exception("Failed to load session info")
            raise RuntimeError(f"Failed to load session info: {e}") from e