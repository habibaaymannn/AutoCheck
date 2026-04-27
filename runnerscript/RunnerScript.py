from __future__ import annotations

import os
import signal
import sys
from pathlib import Path

try:
    import psutil
except ImportError:
    import subprocess
    print("[AutoCheck] psutil not found - installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
    import psutil
    print("[AutoCheck] psutil installed successfully")

from config.ConfigManager import ConfigManager, ConfigParseError, ConfigValidationError
from config.YamlOBJ.System import System
from config.YamlOBJ.Checkpoint import Checkpoint
from AtonomusController.Controller import Controller
from Utilites.logger import setup_logger

PID_FILENAME = ".autocheck.pid"


class RunnerScript:
    """
    Thin orchestrator between the CLI and the Controller.

    run()    → fresh start
    resume() → restore from latest checkpoint, then continue
    stop()   → find running job via PID file, send SIGTERM
    """

    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__, "runner")

    # ------------------------------------------------------------------
    # Public commands (called by CLI)
    # ------------------------------------------------------------------

    def run(
        self,
        config_path: str,
        mode_override: str | None = None,
        save_dir_override: str | None = None,
        validate_only: bool = False,
    ) -> None:
        cm = self._bootstrap(config_path, mode_override, save_dir_override)

        if validate_only:
            print("[AutoCheck] Config is valid.")
            sys.exit(0)

        if self._is_running(cm):
            print("[AutoCheck] A job is already running — stop or resume first.")
            sys.exit(1)

        run_id = cm.get(System).run_id
        self.logger.info(f"Starting fresh run | run_id={run_id}")
        Controller(config=cm, resume=False, run_id=run_id).start_tool()

    def resume(
        self,
        config_path: str,
        mode_override: str | None = None,
        save_dir_override: str | None = None,
    ) -> None:
        cm = self._bootstrap(config_path, mode_override, save_dir_override)

        if self._is_running(cm):
            print("[AutoCheck] A job is already running — stop it before resuming.")
            sys.exit(1)

        run_id = cm.get(System).run_id
        self.logger.info(f"Resuming run | run_id={run_id}")
        Controller(config=cm, resume=True, run_id=run_id).start_tool()

    def stop(
        self,
        config_path: str,
        mode_override: str | None = None,
        save_dir_override: str | None = None,
    ) -> None:
        """
        Runs in a separate process.
        Reads the PID file written by the running Controller and sends SIGTERM.
        The Controller handles the signal: saves checkpoint, deletes PID file, exits.
        """
        cm = self._bootstrap(config_path, mode_override, save_dir_override)

        pid_file = Path(cm.get(Checkpoint).save_dir) / PID_FILENAME

        # no PID file → nothing is running
        if not pid_file.exists():
            print("[AutoCheck] No running job found — nothing to stop.")
            return

        pid = int(pid_file.read_text().strip())

        # stale PID file — process already dead
        if not psutil.pid_exists(pid):
            print(f"[AutoCheck] No running job found (stale PID {pid}) — cleaning up.")
            pid_file.unlink(missing_ok=True)
            return

        print(f"[AutoCheck] Sending stop signal to job (PID {pid})...")
        os.kill(pid, signal.SIGTERM)
        print("[AutoCheck] Stop signal sent — job will save a checkpoint and exit.")
        self.logger.info(f"SIGTERM sent | pid={pid}")

    # ------------------------------------------------------------------
    # Bootstrap — shared by all three commands
    # ------------------------------------------------------------------

    def _bootstrap(
        self,
        config_path: str,
        mode_override: str | None,
        save_dir_override: str | None,
    ) -> ConfigManager:
        # parse config
        cm = ConfigManager()
        try:
            cm.parse(config_path)
        except FileNotFoundError:
            print(f"[AutoCheck] Config not found: {config_path}")
            sys.exit(1)
        except ConfigParseError as e:
            print(f"[AutoCheck] Config parse error: {e}")
            sys.exit(1)

        # apply CLI overrides
        if mode_override:
            from Utilites.enums import ExecutionMode
            cm.get(System).execution_mode = ExecutionMode(mode_override)

        if save_dir_override:
            cm.get(Checkpoint).save_dir = save_dir_override

        # validate
        try:
            cm.validate()
        except ConfigValidationError as e:
            print(f"[AutoCheck] Config validation error: {e}")
            sys.exit(1)

        # validate user program from config exists on disk
        program_path = cm.get(Checkpoint).program_path
        if not os.path.isfile(program_path):
            print(f"[AutoCheck] Program not found: {program_path}")
            sys.exit(1)

        # ensure checkpoint directory exists
        Path(cm.get(Checkpoint).save_dir).mkdir(parents=True, exist_ok=True)

        self.logger.info(
            f"Bootstrap complete | mode={cm.get_mode()} | program={program_path}"
        )
        return cm

    # ------------------------------------------------------------------
    # PID check helper
    # ------------------------------------------------------------------

    def _is_running(self, cm: ConfigManager) -> bool:
        pid_file = Path(cm.get(Checkpoint).save_dir) / PID_FILENAME
        if not pid_file.exists():
            return False
        try:
            pid = int(pid_file.read_text().strip())
            return psutil.pid_exists(pid)
        except (ValueError, OSError):
            return False