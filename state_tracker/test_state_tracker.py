from typing import Any, Mapping, Sequence

import pytest

from state_tracker.Models import TrackedStateSpec
from state_tracker.StateTracker import StateProvider
from state_tracker.HPCStateTracker import HPCStateTracker


class DummyProvider(StateProvider):
    def __init__(self, values: Mapping[str, Any]):
        self.values = dict(values)

    def fetch(self, specs: Sequence[TrackedStateSpec]) -> Mapping[str, Any]:
        return self.values


def test_hpc_tracker_update_from_provider():
    specs = [
        TrackedStateSpec(name="nodes", type_name="int", source="sinfo"),
        TrackedStateSpec(name="healthy", type_name="bool", source="health"),
    ]
    tracker = HPCStateTracker(
        run_id="run-1",
        tracked_states=specs,
        provider=DummyProvider({"nodes": "1", "healthy": "true"}),
    )
    snap = tracker.update_once()
    assert snap.states["nodes"] == 1
    assert snap.states["healthy"] is True
    assert snap.mode == "hpc"


# def test_ml_tracker_update_from_raw_without_provider():
#     specs = [TrackedStateSpec(name="loss", type_name="float", source="train.loss")]
#     tracker = MLStateTracker(run_id="run-2", tracked_states=specs)
#     snap = tracker.update_from_raw({"loss": "0.125"})
#     assert snap.states["loss"] == pytest.approx(0.125)
#     assert snap.mode == "ml"
