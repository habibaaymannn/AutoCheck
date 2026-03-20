from typing import Any, Dict, List

from layers.BaseLayer import BaseLayer


class Provider():
    def __init__(self) -> None:
        self._layers: List[BaseLayer] = []
    def fetch_ckpt(self) -> Dict[str, Any]:
        pass
    def fetch_all(self):
        pass