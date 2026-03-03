from typing import Dict, Any, Optional

from stateTracker.BaseTracker import BaseTracker
from logger import setup_logger

_INT_FIELDS = ("epoch", "global_step", "batch_idx")


class MLStateTracker(BaseTracker):
    def __init__(self, run_id: str, method: str):
        super().__init__(run_id, method)
        self.logger = setup_logger(self.__class__.__name__, run_id)
        self.model_state: Dict[str, Any] = {}
        self.optimizer_state: Dict[str, Any] = {}
        self.scheduler_state: Optional[Dict[str, Any]] = None
        self.epoch: int = 0
        self.global_step: int = 0
        self.batch_idx: int = 0
        self.rng_state: Dict[str, Any] = {}
        self.amp: Dict[str, Any] = {}
        self.logger.info(f"MLStateTracker initialised | run_id={run_id} | method={method}")

    def update_chpnt_method(self) -> None:
        with self.lock:
            if self.provider is None:
                raise RuntimeError("Provider is not set.")
            prov_state: Dict[str, Any] = self.provider.get_state()
            self.epoch = prov_state.get("epoch", self.epoch)
            self.global_step = prov_state.get("global_step", self.global_step)
            self.batch_idx = prov_state.get("batch_idx", self.batch_idx)
            self.logger.info(
                f"Checkpoint fields updated | epoch={self.epoch} "
                f"global_step={self.global_step} batch_idx={self.batch_idx}"
            )

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            state: Dict[str, Any] = {
                "run_id": self.run_id,
                "method": self.method,
                "start_time": self.start_time.isoformat(),
                "model_state": self.model_state,
                "optimizer_state": self.optimizer_state,
                "scheduler_state": self.scheduler_state,
                "epoch": self.epoch,
                "global_step": self.global_step,
                "batch_idx": self.batch_idx,
                "rng_state": self.rng_state,
                "amp": self.amp,
            }
            self.logger.info(f"Snapshot taken | epoch={self.epoch} global_step={self.global_step}")
            return state

    def update_all_from_prov(self) -> None:
        with self.lock:
            if self.provider is None:
                raise RuntimeError("Provider is not set.")
            prov_state: Dict[str, Any] = self.provider.get_state()
            self.model_state = prov_state.get("model_state", {})
            self.optimizer_state = prov_state.get("optimizer_state", {})
            self.scheduler_state = prov_state.get("scheduler_state", None)
            self.epoch = prov_state.get("epoch", self.epoch)
            self.global_step = prov_state.get("global_step", self.global_step)
            self.batch_idx = prov_state.get("batch_idx", self.batch_idx)
            self.rng_state = prov_state.get("rng_state", {})
            self.amp = prov_state.get("amp", {})
            self.logger.info(
                f"All fields updated from provider | epoch={self.epoch} "
                f"global_step={self.global_step} batch_idx={self.batch_idx}"
            )

    def set_all_from_chpnt(self, state: Dict[str, Any]) -> None:
        pass

    def validate(self) -> bool:
        """Return True if all fields are well-formed and safe to snapshot."""
        with self.lock:
            self.logger.debug("validate: checking tracker state integrity")

            checks = {
                "model_state": (self.model_state, dict),
                "optimizer_state": (self.optimizer_state, dict),
                "rng_state": (self.rng_state, dict),
                "amp": (self.amp, dict),
            }
            for field, (value, expected_type) in checks.items():
                if not isinstance(value, expected_type):
                    self.logger.warning(f"Validation failed: '{field}' is {type(value).__name__}, expected dict")
                    return False

            if self.scheduler_state is not None and not isinstance(self.scheduler_state, dict):
                self.logger.warning(
                    f"Validation failed: 'scheduler_state' is {type(self.scheduler_state).__name__}, expected dict or None"
                )
                return False

            for field in _INT_FIELDS:
                value = getattr(self, field)
                if not isinstance(value, int) or value < 0:
                    self.logger.warning(
                        f"Validation failed: '{field}' is {value!r}, expected non-negative int"
                    )
                    return False

            self.logger.info("Validation passed")
            return True
