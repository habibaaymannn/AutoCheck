from provider.Provider import Provider
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from threading import RLock
from datetime import datetime

class BaseTracker(ABC):
    def __init__(self, method: str, program_path: str ,run_id: str = "default"):
        self.run_id: str = run_id
        self.provider: Optional[Provider] = None
        self.lock = RLock()
        self.start_time: datetime = datetime.now()
        self.method: str = method
        self.program_path: str = program_path

    @abstractmethod
    def _init_provider(self):
        pass

    def run_tracer(self):
        self.provider.run_tracer()

    def set_provider(self, provider: Provider) -> None:
        with self.lock:
            self.provider = provider

    @abstractmethod
    def update_ckpt_method(self) -> None:
        pass

    @abstractmethod
    def snapshot(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def update_all_from_prov(self) -> None:
        pass

    @abstractmethod
    def set_all_from_ckpt(self, state: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def validate(self) -> bool:
        pass