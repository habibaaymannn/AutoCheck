import pickle
import threading
from pathlib import Path
from typing import Any, Dict


class StateStore:
  
    def __init__(self, filename: str = "state.pkl") -> None:
        self._data: Dict[str, Any] = {}
        self.filename = Path(filename)
        self._lock = threading.RLock()  # RLock compatible with nested access

        # Load existing state if file exists
        if self.filename.exists():
            self._load()

    def _load(self) -> None:
        """Load state from file."""
        with self._lock:
            try:
                with self.filename.open("rb") as f:
                    self._data = pickle.load(f)
            except Exception:
                self._data = {}  # If corrupted, start fresh

    def _save(self) -> None:
       
        with self._lock:
            with self.filename.open("wb") as f:
                pickle.dump(self._data, f)

    def set(self, key: str, value: Any) -> None:
        
        with self._lock:
            self._data[key] = value
            self._save()

    def get(self, key: str, default: Any = None) -> Any:
        
        with self._lock:
            return self._data.get(key, default)

    def update(self, state_dict: Dict[str, Any]) -> None:
        
        with self._lock:
            self._data.update(state_dict)
            self._save()

    def reset(self) -> None:
       
        with self._lock:
            self._data.clear()
            self._save()

    def get_all(self) -> Dict[str, Any]:
      
        with self._lock:
            return self._data.copy()

    # Optional: support stop_flag for HPCStateTracker
    @property
    def stop_flag(self) -> bool:
        return bool(self.get("stop_flag", False))

    @stop_flag.setter
    def stop_flag(self, value: bool) -> None:
        self.set("stop_flag", value)
