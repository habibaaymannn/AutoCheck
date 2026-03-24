# RunnerScript.py
from __future__ import annotations
import os
import runpy
import sys
import signal
from pathlib import Path
from typing import Optional

import torch

# Cross-platform process checks
try:
    import psutil
except ImportError:
    import subprocess
    print("[AutoCheck] psutil not found — installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
    import psutil
    print("[AutoCheck] psutil installed successfully")

from config.ConfigManager import ConfigManager, ConfigParseError, ConfigValidationError
from config.YamlOBJ.System import System
from config.YamlOBJ.Checkpoint import Checkpoint
from config.YamlOBJ.HPC import HPC
from stateTracker.MLStateTracker import MLStateTracker
from stateTracker.HPCStateTracker import HPCStateTracker
from provider.Provider import Provider
from logger import setup_logger

# -----------------------------
# Minimal stub controller
# -----------------------------
class AutonomousController:
    def __init__(self):
        self._config = None
        self._tracker = None

    def set_config(self, cm):
        self._config = cm
        print(f"[STUB] set_config | mode={cm.mode}")

    def set_state_tracker(self, mode, tracker):
        self._tracker = tracker
        print(f"[STUB] set_state_tracker | mode={mode} | tracker={type(tracker).__name__}")

    def start(self):
        print("[STUB] start() — running user script normally")

# -----------------------------
# RunnerScript
# -----------------------------
class RunnerScript:
    def __init__(self):
        self.logger = setup_logger("RunnerScript", "runner")

    # -------------------------
    # Public API
    # -------------------------
    def run(self, config_path, user_program, mode_override=None, save_dir_override=None,
            validate_only=False, model=None, optimizer=None, scheduler=None,
            global_step=None, epoch=None, batch_idx=None):

        cm = self._bootstrap(config_path, user_program, mode_override, save_dir_override)
        tracker, provider, checkpoint_dir, keep_last = self._setup_checkpoint(
            cm, user_program, model, optimizer, scheduler, global_step, epoch, batch_idx
        )

        if validate_only:
            print("[AutoCheck] config is valid")
            sys.exit(0)

        if self._is_running(cm):
            print("[AutoCheck] a job is already running. Stop or resume first.")
            sys.exit(1)

        controller = self._build_controller(cm, tracker)
        controller.start()

        self._run_with_checkpoint(user_program, tracker, provider, checkpoint_dir, keep_last)

    def resume(self, config_path, user_program, mode_override=None, save_dir_override=None,
               model=None, optimizer=None, scheduler=None,
               global_step=None, epoch=None, batch_idx=None):

        cm = self._bootstrap(config_path, user_program, mode_override, save_dir_override)
        tracker, provider, checkpoint_dir, keep_last = self._setup_checkpoint(
            cm, user_program, model, optimizer, scheduler, global_step, epoch, batch_idx
        )

        controller = self._build_controller(cm, tracker)
        controller.start()

        # Load checkpoint if available
        payload = self._load_checkpoint(checkpoint_dir)
        if payload:
            provider.restore(payload)
            self._print_state("Restoring checkpoint", payload)
        else:
            print("[AutoCheck] No checkpoint found — starting fresh")

        self._run_with_checkpoint(user_program, tracker, provider, checkpoint_dir, keep_last)

    # -------------------------
    # Bootstrap / wiring
    # -------------------------
    def _bootstrap(self, config_path, user_program, mode_override, save_dir_override):
        if not os.path.isfile(user_program):
            print(f"[AutoCheck] Program not found: {user_program}")
            sys.exit(1)

        cm = ConfigManager()
        try:
            cm.parse(config_path)
        except FileNotFoundError:
            print(f"[AutoCheck] Config not found: {config_path}")
            sys.exit(1)
        except ConfigParseError as e:
            print(f"[AutoCheck] Config parse error: {e}")
            sys.exit(1)

        if mode_override:
            system: System = cm.get(System)
            from enums import ExecutionMode
            system.execution_mode = ExecutionMode(mode_override)

        if save_dir_override:
            try:
                checkpoint: Checkpoint = cm.get(Checkpoint)
                checkpoint.save_dir = save_dir_override
            except KeyError:
                pass

        try:
            cm.validate()
        except ConfigValidationError as e:
            print(f"[AutoCheck] Config validation error: {e}")
            sys.exit(1)

        self.logger.info(f"Config loaded | mode={cm.mode}")
        return cm

    def _build_controller(self, cm, tracker):
        controller = AutonomousController()
        controller.set_config(cm)
        controller.set_state_tracker(cm.mode, tracker)
        self.logger.info(f"Controller wired | mode={cm.mode}")
        return controller

    # -------------------------
    # Checkpoint / Provider
    # -------------------------
    def _setup_checkpoint(self, cm, user_program, model=None, optimizer=None, scheduler=None,
                          global_step=None, epoch=None, batch_idx=None):

        system_cfg = cm.get(System)
        checkpoint_cfg = cm.get(Checkpoint)
        checkpoint_dir = Path(checkpoint_cfg.save_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        keep_last = checkpoint_cfg.keep_last

        abs_path = os.path.abspath(user_program)
        method = checkpoint_cfg.method

        # Setup tracker
        if cm.mode == "ml":
            tracker = MLStateTracker(method=method, program_path=abs_path, run_id=system_cfg.run_id)
        elif cm.mode == "hpc":
            hpc_cfg = cm.get(HPC)
            tracker = HPCStateTracker(method=method, program_path=abs_path,
                                      tracked_states=hpc_cfg.tracked_states, scheduler=system_cfg.fram_schd,
                                      run_id=system_cfg.run_id)
        else:
            raise ValueError(f"Unsupported mode {cm.mode}")

        # Provider handles polling / snapshotting
        provider = tracker.provider
        tracker.run_tracer()

        return tracker, provider, checkpoint_dir, keep_last

    def _save_checkpoint(self, snapshot: dict, checkpoint_dir: Path, keep_last: int):
        step = snapshot.get("global_step", 0)
        path = checkpoint_dir / f"checkpoint_{step:08d}.pt"
        tmp = path.with_suffix(".tmp")
        torch.save(snapshot, tmp)
        os.replace(tmp, path)
        self._prune_checkpoints(checkpoint_dir, keep_last)
        print(f"[Checkpoint] saved → {path.name}")
        return path

    def _load_checkpoint(self, checkpoint_dir: Path):
        files = sorted(checkpoint_dir.glob("checkpoint_*.pt"))
        if not files:
            return None
        payload = torch.load(files[-1])
        print(f"[Checkpoint] loaded ← {files[-1].name}")
        return payload

    def _prune_checkpoints(self, checkpoint_dir: Path, keep_last: int):
        files = sorted(checkpoint_dir.glob("checkpoint_*.pt"))
        for old in files[:-keep_last]:
            old.unlink()

    # -------------------------
    # Run wrapper with Ctrl+C handling
    # -------------------------
    def _run_with_checkpoint(self, user_program, tracker, provider, checkpoint_dir, keep_last):
        def handle_sigint(sig, frame):
            print("\n[AutoCheck] Ctrl+C caught — saving checkpoint...")
            snapshot = provider.fetch_all()
            self._save_checkpoint(snapshot, checkpoint_dir, keep_last)
            self._print_state("AutoCheck saved", snapshot)
            sys.exit(0)

        # Attach Ctrl+C handler
        signal.signal(signal.SIGINT, handle_sigint)

        try:
            runpy.run_path(user_program, run_name="__main__")
            snapshot = provider.fetch_all()
            self._save_checkpoint(snapshot, checkpoint_dir, keep_last)
        except Exception as e:
            print(f"\n[AutoCheck] Exception: {e}")
            snapshot = provider.fetch_all()
            self._save_checkpoint(snapshot, checkpoint_dir, keep_last)
            self._print_state("AutoCheck saved due to exception", snapshot)

    # -------------------------
    # State printing
    # -------------------------
    def _print_state(self, label, state: dict):
        print(f"\n[{label}]")
        for k, v in state.items():
            if isinstance(v, (int, float)):
                print(f"  {k:12} = {v}")
            elif isinstance(v, dict):
                print(f"  {k:12} = dict with {len(v)} keys")
            else:
                print(f"  {k:12} = {type(v).__name__}")

    # -------------------------
    # Process / PID helpers
    # -------------------------
    def _is_running(self, cm):
        pid_file = Path(cm.get(Checkpoint).save_dir) / ".autocheck.pid"
        if not pid_file.exists():
            return False
        pid = int(pid_file.read_text())
        return psutil.pid_exists(pid)