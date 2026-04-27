import os
import runpy
import signal
import threading
import time
from datetime import datetime
from pathlib import Path
from Utilites.logger import setup_logger
from checkpointManager.KerasCheckpointManager import KerasCheckpointManager
from checkpointManager.SKLearnCheckpointManager import SKLearnCheckpointManager
from checkpointManager.GenericCheckpointManager import GenericCheckpointManager
from checkpointManager.PyTorchCheckpointManager import PyTorchCheckpointManager
from config.ConfigManager import ConfigManager
from stateTracker.HPCStateTracker import HPCStateTracker
from stateTracker.MLStateTracker import MLStateTracker

TICK_SECONDS = 60  # how often the checkpoint loop wakes up
PID_FILENAME = ".autocheck.pid"


class Controller:
    def __init__(self, config: ConfigManager, resume: bool = False, run_id: str = "default"):
        #set config
        self.config = config
        self.run_id = run_id

        #set tracker without init the provider
        mode = config.get_mode()
        ckpt_config = config.get_ckpt_config()
        method = ckpt_config["method"]
        program_path = ckpt_config["program_path"]
        self.tracker = MLStateTracker(method, program_path, run_id)
        if mode == "hpc":
            states = config.get_tracked_states()
            self.tracker = HPCStateTracker(method, program_path, states, run_id=run_id)

        #ckpt manager
        manager_type = config.get_ckpt_manager_type()
        save_dir = ckpt_config.get("save_dir")
        max_to_keep = ckpt_config.get("keep_last")
        if manager_type == "pytorch":
            self.ckptManager = PyTorchCheckpointManager(save_dir, max_to_keep)
        elif manager_type == "tensorflow":
            self.ckptManager = KerasCheckpointManager(save_dir, max_to_keep)
        elif manager_type == "sklearn":
            self.ckptManager = SKLearnCheckpointManager(save_dir, max_to_keep)
        else:
            self.ckptManager = GenericCheckpointManager(save_dir, max_to_keep)

        # start timer
        self.start_time = datetime.now()
        #max_session_time & safety_buffer & interval & ckpt_method & Logger & resume & save_dir
        self.ckpt_method = method
        self.max_session_time = ckpt_config["max_session_time"]
        self.safety_buffer_seconds = ckpt_config["safety_buffer_seconds"]
        self.interval = ckpt_config["interval"]
        self.resume = resume
        self.save_dir = save_dir
        self.program_path = program_path
        self.logger = setup_logger(self.__class__.__name__, run_id)

        # internal state for the checkpoint loop
        self._last_ckpt_time: float = time.monotonic()  # for time-based method
        self._last_ckpt_value: int = 0  # for step/epoch-based method
        self._script_thread: threading.Thread | None = None
        self._pid_file: Path = Path(save_dir) / PID_FILENAME


        self.logger.info(
            f"Controller initialised | run_id={run_id} | mode={mode} | "
            f"ckpt_manager={manager_type} | method={method}"
        )

    def start_tool(self):
        # 1. init provider (creates TraceLayer, but user script not running yet)
        self.tracker.init_provider()
        self.tracker.run_tracer()  # sys.settrace installed — safe before runpy

        # 2. restore from checkpoint if resuming (provider exists now)
        if self.resume:
            state = self.ckptManager.load_checkpoint(self.save_dir)
            if state:
                self.tracker.set_all_from_ckpt(state)
                self.logger.info("Checkpoint restored | resuming from saved state")
            else:
                self.logger.warning("Resume requested but no checkpoint found | starting fresh")

        # 3. launch user script in a background thread
        self._script_thread = threading.Thread(
            target=self._run_user_script,
            name="user-script",
            daemon=True,  # dies automatically if main thread is killed
        )
        self._script_thread.start()
        self.logger.info(f"User script thread started | program={self.config.get_ckpt_config()['program_path']}")

        # 4. run checkpoint loop on main thread (blocks until script finishes or we force-exit)
        self._checkpoint_loop()

    # ------------------------------------------------------------------
    # Stop — called internally only (safety buffer / signal handler)
    # External CLI stop sends SIGTERM via the PID file → _handle_signal → _stop
    # ------------------------------------------------------------------

    def _stop(self, reason: str = "manual"):
        self.logger.info(f"Stop triggered | reason={reason}")
        self._take_checkpoint(reason=f"stop:{reason}")
        self._delete_pid_file()
        self.logger.info("Shutdown complete")
        os.kill(os.getpid(), signal.SIGKILL)  # force-kill everything including user thread

    # ------------------------------------------------------------------
    # Signal handler (SIGTERM / SIGINT)
    # ------------------------------------------------------------------

    def _handle_signal(self, signum: int, frame) -> None:
        sig_name = signal.Signals(signum).name
        self.logger.warning(f"Signal received | signal={sig_name}")
        self._stop(reason=sig_name)

    # ------------------------------------------------------------------
    # PID file helpers
    # ------------------------------------------------------------------

    def _write_pid_file(self) -> None:
        try:
            self._pid_file.write_text(str(os.getpid()))
            self.logger.info(f"PID file written | path={self._pid_file} | pid={os.getpid()}")
        except OSError as e:
            self.logger.error(f"Failed to write PID file | reason={e}")

    def _delete_pid_file(self) -> None:
        try:
            if self._pid_file.exists():
                self._pid_file.unlink()
                self.logger.info(f"PID file removed | path={self._pid_file}")
        except OSError as e:
            self.logger.error(f"Failed to remove PID file | reason={e}")

    def _run_user_script(self):
        try:
            self.logger.info(f"Running user script | path={self.program_path}")
            runpy.run_path(self.program_path, run_name="__main__")
            self.logger.info("User script finished normally")
        except SystemExit as e:
            # SystemExit is normal (e.g. sys.exit(0)) — not an error
            self.logger.info(f"User script exited | code={e.code}")
        except Exception as e:
            self.logger.error(f"User script raised an exception | reason={e}", exc_info=True)

    def _checkpoint_loop(self):
        self.logger.info(
            f"Checkpoint loop started | method={self.ckpt_method} | "
            f"interval={self.interval} | max_session_time={self.max_session_time}s | "
            f"safety_buffer={self.safety_buffer_seconds}s"
        )

        while self._script_thread.is_alive():
            time.sleep(TICK_SECONDS)

            elapsed = (datetime.now() - self.start_time).total_seconds()

            # ── SAFETY BUFFER — highest priority, always checked first ──
            if elapsed >= (self.max_session_time - self.safety_buffer_seconds):
                self.logger.warning(
                    f"Safety buffer triggered | elapsed={elapsed:.0f}s | "
                    f"limit={self.max_session_time - self.safety_buffer_seconds:.0f}s"
                )
                self._stop(reason="safety_buffer")
                return  # unreachable but explicit

            # ── NORMAL CHECKPOINTING ────────────────────────────────────
            if self.ckpt_method == "time":
                self._check_time_based()
            else:
                self._check_value_based()

        # script finished normally — take a final checkpoint
        self.logger.info("User script finished — taking final checkpoint")
        self._take_checkpoint(reason="final")
        self._delete_pid_file()

    def _check_time_based(self):
        now = time.monotonic()
        if (now - self._last_ckpt_time) >= self.interval:
            self._take_checkpoint(reason="interval")
            self._last_ckpt_time = now

    def _check_value_based(self):
        """
        Polls the tracker for the current value of the tracked variable
        (epoch, global_step, iteration, etc.) and checkpoints whenever
        it has advanced by at least self.interval units since the last checkpoint.
        """
        try:
            self.tracker.update_ckpt_method()
        except Exception as e:
            self.logger.error(f"Failed to poll tracker | reason={e}")
            return

        # get the current value of the tracked variable from the tracker
        current_value = self._get_tracked_value()
        if current_value is None:
            return

        if (current_value - self._last_ckpt_value) >= self.interval:
            self._take_checkpoint(reason=f"{self.ckpt_method}={current_value}")
            self._last_ckpt_value = current_value

    def _get_tracked_value(self) -> int | None:
        """
        Reads the current value of the method variable from the tracker.
        Supports epoch, global_step, batch_idx (ML) and iteration (HPC).
        """
        method = self.ckpt_method
        value = getattr(self.tracker, method, None)
        if value is None:
            self.logger.warning(f"Tracked variable '{method}' not found on tracker")
        return value

    def _take_checkpoint(self, reason: str = ""):
        try:
            self.logger.info(f"Taking checkpoint | reason={reason}")
            snapshot = self.tracker.snapshot()
            self.ckptManager.save_checkpoint(snapshot, self.save_dir)
            self.logger.info(f"Checkpoint saved | reason={reason}")
        except Exception as e:
            self.logger.error(f"Checkpoint failed | reason={reason} | error={e}", exc_info=True)