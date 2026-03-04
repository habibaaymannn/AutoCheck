from typing import Dict, Any, Optional

from provider import Provider
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

    def set_provider(self, provider: Provider) -> None:
        self.provider = provider
        self.logger.info(
            f"[SET_PROVIDER] | provider={type(provider).__name__}"
        )

    def update_chpnt_method(self) -> None:
        with self.lock:
            if self.provider is None:
                self.logger.error("[UPDATE_METHOD] | Provider not set")
                raise RuntimeError("Provider is not set.")

            self.logger.debug("[UPDATE_METHOD] | Fetching provider state")
            try:
                prov_state = self.provider.fetch_ckpnt()
                if not isinstance(prov_state, dict):
                    self.logger.error(
                        f"[UPDATE_METHOD] | Invalid provider state type | "
                        f"type={type(prov_state).__name__}"
                    )
                    raise TypeError("Provider checkpoint state must be a dictionary.")

                required_fields = ["epoch", "global_step", "batch_idx"]
                missing = [f for f in required_fields if f not in prov_state]
                if missing:
                    self.logger.error(f"[UPDATE_METHOD] Missing checkpoint fields | missing={missing}")
                    raise KeyError(f"Missing checkpoint fields: {missing}")

                old_epoch = self.epoch
                old_step = self.global_step
                old_batch = self.batch_idx

                self.epoch = prov_state['epoch']
                self.global_step = prov_state["global_step"]
                self.batch_idx = prov_state["batch_idx"]
                self.logger.info(
                    f"[UPDATE_METHOD] | Checkpoint counters updated | "
                    f"epoch {old_epoch}->{self.epoch} | "
                    f"global_step {old_step}->{self.global_step} | "
                    f"batch_idx {old_batch}->{self.batch_idx}"
                )
            except Exception as e:
                self.logger.error(f"[UPDATE_METHOD] | Failed to update checkpoint fields | reason={e}")
                raise

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
            self.logger.info(f"[SNAPSHOT] | Snapshot taken | epoch={self.epoch} global_step={self.global_step}")
            return state

    def update_all_from_prov(self) -> None:
        with self.lock:
            if self.provider is None:
                self.logger.error(f"[UPDATE_ALL] | Provider not set")
                raise RuntimeError("Provider is not set.")

            self.logger.debug(f"[UPDATE_ALL] | Pulling full state from provider")

            try:
                prov_state = self.provider.fetch_all()
                if not isinstance(prov_state, dict):
                    self.logger.error(
                        f"[UPDATE_ALL] | Invalid provider state type | "
                        f"type={type(prov_state).__name__}"
                    )
                    raise TypeError("Provider state must be a dictionary.")
                required_fields = ["model_state", "optimizer_state", "epoch", "global_step", "batch_idx"]
                missing = [f for f in required_fields if f not in prov_state]
                if missing:
                    self.logger.error(f"[UPDATE_ALL] | Missing checkpoint fields | missing={missing}")
                    raise KeyError(f"Missing checkpoint fields: {missing}")

                old_epoch = self.epoch
                old_step = self.global_step
                old_batch = self.batch_idx

                self.model_state = prov_state["model_state"]
                self.optimizer_state = prov_state["optimizer_state"]
                self.scheduler_state = prov_state.get("scheduler_state", None)
                self.epoch = prov_state["epoch"]
                self.global_step = prov_state["global_step"]
                self.batch_idx = prov_state["batch_idx"]
                self.rng_state = prov_state.get("rng_state", {})
                self.amp = prov_state.get("amp", {})
                self.logger.info(
                    f"[UPDATE_ALL] | All fields updated from provider | "
                    f"epoch {old_epoch}->{self.epoch} | "
                    f"global_step {old_step}->{self.global_step} | "
                    f"batch_idx {old_batch}->{self.batch_idx}"
                )
            except Exception as e:
                self.logger.error(f"[UPDATE_ALL] | Failed to update all fields | reason={e}")
                raise e

    def set_all_from_chpnt(self, state: Dict[str, Any]) -> None:
        with self.lock:
            if not isinstance(state, dict):
                self.logger.error(f"[RESTORE] | Checkpoint restore failed | invalid type={type(state).__name__}")
                raise TypeError("Checkpoint state must be a dictionary.")

            try:
                required_fields = ["model_state", "optimizer_state", "epoch", "global_step", "batch_idx"]
                missing = [f for f in required_fields if f not in state]
                if missing:
                    self.logger.error(f"Checkpoint restore failed | missing fields={missing}")
                    raise KeyError(f"Checkpoint missing required fields: {missing}")

                self.model_state = state["model_state"]
                self.optimizer_state = state["optimizer_state"]
                self.scheduler_state = state.get("scheduler_state", None)
                self.epoch = state["epoch"]
                self.global_step = state["global_step"]
                self.batch_idx = state["batch_idx"]
                self.rng_state = state.get("rng_state", {})
                self.amp = state.get("amp", {})

                if not self.validate():
                    self.logger.error("[RESTORE] | Checkpoint restore failed validation")
                    raise ValueError("Restored checkpoint failed validation.")

                self.logger.info(
                    f"[RESTORE] | State restored from checkpoint | "
                    f"epoch={self.epoch} global_step={self.global_step} "
                    f"batch_idx={self.batch_idx}"
                )
            except Exception as e:
                self.logger.error(f"[RESTORE] | Failed to restore checkpoint: {e}")
                raise

    def validate(self) -> bool:
        """Return True if all fields are well-formed and safe to snapshot."""
        with self.lock:
            self.logger.debug("[validate] | checking tracker state integrity")

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
