from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Mapping

@dataclass(frozen=True)
class TrackedStateSpec:
    name:str
    type:str
    source: str

    def normalized_type(self)->str:
        return self.type_name.strip().lower()
    

@dataclass(frozen=True)
class StateSnapshot:
    run_id: str
    mode: str
    states:Dict[str,Any]
    captured_at_utc: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    schema_version: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "mode": self.mode,
            "captured_at_utc": self.captured_at_utc,
            "states": dict(self.states),
            "metadata": dict(self.metadata),
        }
    
    @staticmethod
    def from_mapping(data: Mapping[str, Any]) -> "StateSnapshot":
        return StateSnapshot(
            run_id=str(data["run_id"]),
            mode=str(data["mode"]),
            states=dict(data.get("states", {})),
            captured_at_utc=str(data["captured_at_utc"]),
            schema_version=int(data.get("schema_version", 1)),
            metadata=dict(data.get("metadata", {})),
        )