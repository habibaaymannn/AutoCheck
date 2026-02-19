from YamlObj import YamlObj
from dataclasses import dataclass
from System import System

@dataclass
class ML(YamlObj):
    name : str
    ml_system: System
    def __init__(self, name, ml_system):
        self.name = name
        self.ml_system = ml_system
        if self.validate():
            pass

    def validate(self) -> bool:
        pass

