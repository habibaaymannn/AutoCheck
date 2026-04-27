from typing import Dict, Any

from config.YamlOBJ.YamlObj import YamlObj
from dataclasses import dataclass
from Utilites.enums import CheckpointMethod

@dataclass
class Checkpoint(YamlObj):
    method: str
    interval: int
    max_session_time: int
    save_dir: str
    program_path: str
    safety_buffer_seconds: int = 15
    keep_last: int = 3

    def get_integration(self) -> Dict[str, Any]:
        return {"method":self.method,
                "interval":self.interval, "max_session_time":self.max_session_time,
                "save_dir":self.save_dir, "safety_buffer_seconds":self.safety_buffer_seconds,
                "keep_last":self.keep_last, "program_path":self.program_path}

    def validate(self) -> bool:
        if not isinstance(self.method,str):
            raise ValueError("method must be a string")
        self.method=self.method.lower()

        allowed_methods={e.value for e in CheckpointMethod}
        if self.method not in allowed_methods:
            raise ValueError(
                f"Invalid checkpoint method '{self.method}'. "
                f"Allowed: {allowed_methods}"
            )

        if self.interval <= 0:
            raise ValueError("interval must be > 0")

        if self.max_session_time <= 0:
            raise ValueError("max_session_time must be > 0")

        if self.safety_buffer_seconds < 0:
            raise ValueError("safety_buffer_seconds must be >= 0")
        # at least 1
        if self.keep_last < 1:
            raise ValueError("keep_last must be >= 1")

        if not self.save_dir:
            raise ValueError("save_dir is required")

        if not self.program_path:
            raise ValueError("program_path is required")

        return True
