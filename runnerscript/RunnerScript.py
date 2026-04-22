from __future__ import annotations

import os
import runpy
import signal
import sys
from pathlib import Path

import torch

from checkpointManager.KerasCheckpointManager import KerasCheckpointManager

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
from config.YamlOBJ.HPC import HPC
from stateTracker.MLStateTracker import MLStateTracker
from stateTracker.HPCStateTracker import HPCStateTracker
from logger import setup_logger


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
        print("[STUB] start() - running user script normally")


class RunnerScript:
    def __init__(self):
        self.logger = setup_logger("RunnerScript", "runner")

    def run(
        self,
        config_path,
        user_program,
        mode_override=None,
        save_dir_override=None,
        validate_only=False,
        model=None,
        optimizer=None,
        scheduler=None,
        global_step=None,
        epoch=None,
        batch_idx=None,
    ):
        cm, checkpoint_dir, keep_last = self._bootstrap(
            config_path, user_program, mode_override, save_dir_override
        )
        tracker, provider, checkpoint_manager = self._setup_checkpoint(
            cm, user_program
        )

        if validate_only:
            print("[AutoCheck] config is valid")
            sys.exit(0)

        if self._is_running(cm):
            print("[AutoCheck] a job is already running. Stop or resume first.")
            sys.exit(1)

        controller = self._build_controller(cm, tracker)
        controller.start()

        payload = self._load_checkpoint(checkpoint_dir, checkpoint_manager)
        if payload:
            provider.restore(payload)
            self._print_state("Auto-resume checkpoint", payload)
        else:
            print("[AutoCheck] No checkpoint found - starting fresh")

        self._run_with_checkpoint(
            user_program,
            provider,
            checkpoint_dir,
            keep_last,
            checkpoint_manager,
        )

    def resume(
            self,
            config_path,
            user_program,
            mode_override=None,
            save_dir_override=None,
            model=None,
            optimizer=None,
            scheduler=None,
            global_step=None,
            epoch=None,
            batch_idx=None,
    ):
        cm, checkpoint_dir, keep_last = self._bootstrap(
            config_path, user_program, mode_override, save_dir_override
        )
        tracker, provider, checkpoint_manager = self._setup_checkpoint(
            cm, user_program
        )

        controller = self._build_controller(cm, tracker)
        controller.start()

        payload = self._load_checkpoint(checkpoint_dir, checkpoint_manager)
        if payload:
            provider.restore(payload)
            self._print_state("Restoring checkpoint", payload)
        else:
            print("[AutoCheck] No checkpoint found - starting fresh")

        self._run_with_checkpoint(
            user_program,
            provider,
            checkpoint_dir,
            keep_last,
            checkpoint_manager,
        )

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

        checkpoint_cfg = cm.get(Checkpoint)
        checkpoint_dir = Path(checkpoint_cfg.save_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        keep_last = checkpoint_cfg.keep_last

        self.logger.info(f"Config loaded | mode={cm.mode}")
        return cm, checkpoint_dir, keep_last

    def _build_controller(self, cm, tracker):
        controller = AutonomousController()
        controller.set_config(cm)
        controller.set_state_tracker(cm.mode, tracker)
        self.logger.info(f"Controller wired | mode={cm.mode}")
        return controller

    def _setup_checkpoint(
            self,
            cm,
            user_program,
    ):
        system_cfg = cm.get(System)
        checkpoint_cfg = cm.get(Checkpoint)
        abs_path = os.path.abspath(user_program)
        method = checkpoint_cfg.method

        if cm.mode == "ml":
            tracker = MLStateTracker(method=method, program_path=abs_path, run_id=system_cfg.run_id)
        elif cm.mode == "hpc":
            hpc_cfg = cm.get(HPC)
            tracker = HPCStateTracker(
                method=method,
                program_path=abs_path,
                tracked_states=hpc_cfg.tracked_states,
                scheduler=system_cfg.fram_schd,
                run_id=system_cfg.run_id,
            )
        else:
            raise ValueError(f"Unsupported mode {cm.mode}")

        provider = tracker.provider
        tracker.run_tracer()

        checkpoint_manager = None

        if cm.mode == "ml":
            framework = str(system_cfg.fram_schd).lower()
            if framework in ["keras", "tf", "tensorflow"]:
                checkpoint_manager = KerasCheckpointManager(
                    checkpoint_dir=checkpoint_cfg.save_dir,
                    max_to_keep=checkpoint_cfg.keep_last,
                )
        return tracker, provider, checkpoint_manager

    def _save_checkpoint(self, snapshot: dict, checkpoint_dir: Path, keep_last: int, checkpoint_manager=None):
        if checkpoint_manager is not None:
            payload = dict(snapshot)
            saved_path = checkpoint_manager.save_checkpoint(payload, str(checkpoint_dir))
            checkpoint_manager.save_session_info(
                str(checkpoint_dir),
                checkpoint_path=saved_path,
            )
            path = Path(saved_path)
            print(f"[Checkpoint] saved -> {path.name}")
            return path

        step = snapshot.get("global_step", 0)
        path = checkpoint_dir / f"checkpoint_{step:08d}.pt"
        tmp = path.with_suffix(".tmp")
        torch.save(snapshot, tmp)
        os.replace(tmp, path)
        self._prune_checkpoints(checkpoint_dir, keep_last)
        print(f"[Checkpoint] saved -> {path.name}")
        return path

    def _load_checkpoint(self, checkpoint_dir: Path, checkpoint_manager=None):
        if checkpoint_manager is not None:
            try:
                payload = checkpoint_manager.load_checkpoint(str(checkpoint_dir))
            except (RuntimeError, FileNotFoundError):
                return None
            version = payload.get("checkpoint_version")
            if version is not None:
                print(f"[Checkpoint] loaded <- v{version}")
            else:
                print("[Checkpoint] loaded <- latest")
            return payload

        files = sorted(checkpoint_dir.glob("checkpoint_*.pt"))
        if not files:
            return None
        payload = torch.load(files[-1])
        print(f"[Checkpoint] loaded <- {files[-1].name}")
        return payload

    def _prune_checkpoints(self, checkpoint_dir: Path, keep_last: int):
        files = sorted(checkpoint_dir.glob("checkpoint_*.pt"))
        for old in files[:-keep_last]:
            old.unlink()

    def _run_with_checkpoint(self, user_program, provider, checkpoint_dir, keep_last, checkpoint_manager=None):
        user_program_abs = os.path.abspath(user_program)

        def handle_sigint(sig, frame):
            print("\n[AutoCheck] Ctrl+C caught - saving checkpoint...")
            snapshot = provider.fetch_all()
            self._save_checkpoint(snapshot, checkpoint_dir, keep_last, checkpoint_manager)
            self._print_state("AutoCheck saved", snapshot)
            sys.exit(0)

        signal.signal(signal.SIGINT, handle_sigint)

        try:
            runpy.run_path(user_program_abs, run_name="__main__")
            snapshot = provider.fetch_all()
            self._save_checkpoint(snapshot, checkpoint_dir, keep_last, checkpoint_manager)
        except Exception as e:
            print(f"\n[AutoCheck] Exception: {e}")
            snapshot = provider.fetch_all()
            self._save_checkpoint(snapshot, checkpoint_dir, keep_last, checkpoint_manager)
            self._print_state("AutoCheck saved due to exception", snapshot)

    def _print_state(self, label, state: dict):
        print(f"\n[{label}]")
        for k, v in state.items():
            if isinstance(v, (int, float)):
                print(f"  {k:12} = {v}")
            elif isinstance(v, dict):
                print(f"  {k:12} = dict with {len(v)} keys")
            else:
                print(f"  {k:12} = {type(v).__name__}")

    def _is_running(self, cm):
        pid_file = Path(cm.get(Checkpoint).save_dir) / ".autocheck.pid"
        if not pid_file.exists():
            return False
        pid = int(pid_file.read_text())
        return psutil.pid_exists(pid)
