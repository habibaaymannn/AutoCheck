from dataclasses import dataclass
from AutoCheck.config.YamlObj import YamlObj


@dataclass
class Notify(YamlObj):
    email: str
    on_failure: bool
    on_checkpoint: bool

    def validate(self) -> bool:

        # Basic email validation: must contain "@"
        if not self.email or "@" not in self.email:
            raise ValueError(f"Invalid email address: '{self.email}'")
            # make sure that boolean flags are actually bool
        if not isinstance(self.on_failure, bool):
            raise ValueError("on_failure must be a boolean value")

        if not isinstance(self.on_checkpoint, bool):
            raise ValueError("on_checkpoint must be a boolean value")

        return True
