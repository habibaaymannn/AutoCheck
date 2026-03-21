from typing import Any, Dict, List

from layers.SignalLayer import SignalLayer
from layers.TraceLayer import TraceLayer
from logger import setup_logger
from layers.BaseLayer import BaseLayer


class Provider:
    """
    Aggregator that sits between the layers and the tracker.

    Holds a list of BaseLayer instances and fans out fetch calls to
    them, merging the results into a single dict.

    Constructed by RunnerScript._build_provider() with the appropriate
    layer list for the current execution mode:

        ML mode:
            provider = Provider(run_id)
            provider.register_layers([TraceLayer(...), SignalLayer(...)])

        HPC mode:
            provider = Provider(run_id)
            provider.register_layers([SignalLayer(...)])

    The provider is then injected into the tracker:
        tracker.set_provider(provider)

    Method contract
    ---------------
    fetch_ckpt()  -- calls layer.poll() on every layer.
                     Returns only lightweight scalar values.
                     Safe to call frequently (no .state_dict() calls).

    fetch_all()   -- calls layer.snapshot() on every layer.
                     Returns scalars AND serialised object states.
                     Called only when a checkpoint is actually triggered.

    restore(snap) -- fans out layer.restore(snap) to every layer.
                     Called once on resume before the user script runs.
    """
    def __init__(self, program_path: str, ckpt_method:str, poll:list[str], snapshot:list[str],  run_id: str = "default") -> None:
        self._layers: List[BaseLayer] = []
        self.program_path = program_path
        self.ckpt_method = ckpt_method
        self.poll = poll
        self.snapshot = snapshot
        self.run_id = run_id
        self.logger = setup_logger(self.__class__.__name__, run_id)

        # init trace layer & signal layer
        trace_layer = TraceLayer(self.program_path, self.poll, self.snapshot, self.run_id)
        signal_layer = SignalLayer(self.poll, self.snapshot)
        self._register_layer(trace_layer)
        self._register_layer(signal_layer)


    def fetch_ckpt(self):
        result : Dict[str, Any] = {}
        for layer in self._layers:
            if isinstance(layer, TraceLayer):
                vars = layer.poll()
                for var in self.poll:
                    if var in vars:
                        result[var] = vars[var]
                if not result :
                    self.logger.error(f"[FETCH_CKPT] | Couldn't find the checkpoint variables {self.poll} in the program")
                    raise ValueError(f"Couldn't find the checkpoint variable in the program")
                else:
                    self.logger.info(f"[FETCH_CKPT] | Found the checkpoint variables in the program: {self.poll}")
            else:
                continue
        return result

    def fetch_all(self):
        result: Dict[str, Any] = {}
        for layer in self._layers:
            if isinstance(layer, TraceLayer):
                vars = layer.snapshot()
                filtered = self._filter(vars)
                for var in self.poll:
                    if var in filtered:
                        result[var] = filtered[var]
                for var in self.snapshot:
                    if var in filtered:
                        result[var] = filtered[var]
            else:
                continue

        return result

    def restore(self, state: Dict[str, Any]):
        for layer in self._layers:
            if isinstance(layer, TraceLayer):
                layer.restore(state)
        

    def run_tracer(self):
        for layer in self._layers:
            if not layer.is_active():
                layer.attach()

    def _register_layer(self, layer: BaseLayer) -> None:
        self._layers.append(layer)

    @staticmethod
    def _filter(values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Strip entries whose value is None.

        SignalLayer always returns {} from poll() and snapshot().
        None values from partially-initialised layers are also stripped
        so trackers never receive a None where they expect a real value.
        """
        return {k: v for k, v in values.items() if v is not None}