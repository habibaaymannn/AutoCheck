from abc import ABC, abstractmethod


class YamlObj(ABC):

    @abstractmethod
    def validate(self) -> bool:
        pass
