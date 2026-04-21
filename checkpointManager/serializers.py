"""
Pluggable serialization strategies for GenericCheckpointManager.

Each serializer is a self-contained object that knows how to write and read
a checkpoint payload to/from disk.  GenericCheckpointManager accepts any
object that satisfies the Serializer interface, so language adapters
(Java, C++, R, Julia …) can plug in their own without touching the manager.

"""


from __future__ import annotations

import abc
import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


class Serializer(abc.ABC):
    """
    Interface for all serializers.
    """

    extension: str = ""

    @abc.abstractmethod
    def dump(self, payload: Dict[str, Any], path: Path) -> None:
        pass

    @abc.abstractmethod
    def load(self, path: Path) -> Dict[str, Any]:
        pass

    def __repr__(self) -> str:
        return f"{type(self).__name__}(extension={self.extension!r})"


# =========================
# Joblib Serializer
# =========================
class JobLibSerializer(Serializer):
    extension = ".pkl"

    def __init__(self, compress: int = 3) -> None:
        self.compress = compress

    def dump(self, payload: Dict[str, Any], path: Path) -> None:
        import joblib
        joblib.dump(payload, path, compress=self.compress)

    def load(self, path: Path) -> Dict[str, Any]:
        import joblib
        return joblib.load(path)


# =========================
# JSON Serializer
# =========================
class _ExtendedEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        try:
            import numpy as np

            if isinstance(obj, np.ndarray):
                return {
                    "__ndarray__": True,
                    "data": obj.tolist(),
                    "dtype": str(obj.dtype),
                }
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)

        except ImportError:
            pass

        return super().default(obj)


def _extended_decoder(dct: Dict[str, Any]) -> Any:
    if dct.get("__ndarray__"):
        try:
            import numpy as np
            return np.array(dct["data"], dtype=dct["dtype"])
        except ImportError:
            pass
    return dct


class JSONSerializer(Serializer):
    extension = ".json"

    def __init__(self, indent: int | None = 2) -> None:
        self.indent = indent

    def dump(self, payload: Dict[str, Any], path: Path) -> None:
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=self.indent, cls=_ExtendedEncoder)

    def load(self, path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh, object_hook=_extended_decoder)


# =========================
# MsgPack Serializer
# =========================
class MsgPackSerializer(Serializer):
    extension = ".msgpack"

    def __init__(self, use_bin_type: bool = True) -> None:
        self.use_bin_type = use_bin_type

    def dump(self, payload: Dict[str, Any], path: Path) -> None:
        try:
            import msgpack
        except ImportError as exc:
            raise ImportError("Install msgpack: pip install msgpack") from exc

        with path.open("wb") as fh:
            fh.write(msgpack.packb(payload, use_bin_type=self.use_bin_type))

    def load(self, path: Path) -> Dict[str, Any]:
        try:
            import msgpack
        except ImportError as exc:
            raise ImportError("Install msgpack: pip install msgpack") from exc

        with path.open("rb") as fh:
            return msgpack.unpackb(fh.read(), raw=False)