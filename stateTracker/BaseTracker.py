from provider import Provider
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from threading import RLock
from datetime import datetime

class BaseTracker(ABC):
    def __init__(self, run_id: str, method: str):
        self.run_id: str = run_id
        self.provider: Optional[Provider] = None
        self.lock = RLock()
        self.start_time: datetime = datetime.now()
        self.method: str = method

    def set_provider(self, provider: Provider) -> None:
        with self.lock:
            self.provider = provider

    @abstractmethod
    def update_chpnt_method(self) -> None:
        pass

    @abstractmethod
    def snapshot(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def update_all_from_prov(self) -> None:
        pass

    @abstractmethod
    def set_all_from_chpnt(self, state: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def validate(self) -> bool:
        pass