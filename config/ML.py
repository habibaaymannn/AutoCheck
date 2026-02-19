from dataclasses import dataclass
from AutoCheck.config.System import System
from AutoCheck.config.YamlObj import YamlObj


@dataclass
class ML(YamlObj):
    name: str
    ml_system: System

    def validate(self) -> bool:
        if not self.name:
            raise ValueError("name is required")

        self.ml_system.validate()
        return True
