from YamlObj import YamlObj
from dataclasses import dataclass


@dataclass
class Checkpoint(YamlObj):
    method: str
    interval: int
    max_session_time: int
    save_dir: str
    safety_buffer_seconds: int = 15
    keep_last: int = 3

    VALID_METHODS = {"time", "iteration", "epoch", "batch", "step"}

    def validate(self) -> bool:
        if self.method not in self.VALID_METHODS:
            raise ValueError(
                f"Invalid checkpoint method '{self.method}'. "
                f"Allowed: {self.VALID_METHODS}"
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

        return True
