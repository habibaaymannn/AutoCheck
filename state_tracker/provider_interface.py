from abc import ABC, abstractmethod
from typing import Any


class StateProvider(ABC):

    @abstractmethod
    def collect_state(self) -> Any:
     #return the states as a dictionary
        pass

    @abstractmethod
    def restore_state(self) -> Any:
# restores the states 
        pass
