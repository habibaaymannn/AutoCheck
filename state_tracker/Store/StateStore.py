from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from state_tracker.Models import StateSnapshot
from state_tracker.errors import StoreContractError


# Abstract Store Interface
class StateStore(ABC):
    """
    Persistence contract for saving/loading StateSnapshot objects.
    """

    @abstractmethod
    def save(self, snapshot: StateSnapshot) -> None:
        pass

    @abstractmethod
    def load(self) -> Optional[StateSnapshot]:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass
