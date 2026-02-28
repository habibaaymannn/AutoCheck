from abc import ABC, abstractmethod


class StateProvider(ABC):

    @abstractmethod
    def collect_state(self) -> dict:
     #return the states as a dictionary
        pass

    @abstractmethod
    def restore_state(self, state: dict):
# restores the states 
        pass