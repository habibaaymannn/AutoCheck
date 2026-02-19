from YamlObj import YamlObj
from dataclasses import dataclass, field

@dataclass
class Notify(YamlObj):
    def __init__(self, email: str, on_failure: bool, on_checkpoint: bool):
        self.email = email
        self.on_failure = on_failure
        self.on_checkpoint = on_checkpoint
        self.validate()

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
