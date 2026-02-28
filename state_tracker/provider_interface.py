from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class StateProvider(ABC):
    """Interface for components that can provide and restore namespaced state."""

    @property
    @abstractmethod
    def namespace(self) -> str:
        """Unique namespace used in the global state store."""

    @abstractmethod
    def collect_state(self) -> Dict[str, Any]:
        """Return the provider state as a plain dictionary."""

    def restore_state(self, state: Dict[str, Any]) -> None:
        """Restore provider state from a dictionary.

        Providers may override this when restore behavior is needed.
        """


def sync_provider_state(provider: StateProvider, store: "StateTracker") -> None:
    """Push provider state into the shared store."""
    store.bulk_update(provider.namespace, provider.collect_state())
