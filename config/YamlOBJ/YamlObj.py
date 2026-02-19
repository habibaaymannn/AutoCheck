from abc import ABC, abstractmethod


class YamlObj(ABC):

    def __post_init__(self):
        self.validate()

    @abstractmethod
    def validate(self) -> bool:
        pass
