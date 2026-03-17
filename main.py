"""
main.py
Entry point for testing AutoCheck with mlTrainingModel.py.

Uses:
  - ConfigManager    → reads config.yaml
  - TraceLayer       → watches mlTrainingModel.py
  - MLStateTracker   → holds the state
  - CheckpointManager (torch.save/load) → saves to save_dir from config

Run 1: python main.py
  → fresh start, press Ctrl+C to pause
  → checkpoint saved to ./checkpoints/

Run 2: python main.py
  → detects checkpoint, restores all values, resumes training
"""
from __future__ import annotations

from stateTracker.MLStateTracker import MLStateTracker

import os
import sys
import runpy
from pathlib import Path

import torch

# ── project imports ───────────────────────────────────────────────────
from config.ConfigManager import ConfigManager, ConfigParseError, ConfigValidationError
from config.YamlOBJ.Checkpoint import Checkpoint
from config.YamlOBJ.System import System
from layers.TraceLayer import TraceLayer

# ── 1. load and validate config ───────────────────────────────────────
cm = ConfigManager()
try:
    cm.parse("config.yaml")
    cm.validate()
    print("✓ config loaded")
except (ConfigParseError, ConfigValidationError) as e:
    print(f"❌ [CONFIG ERROR] {e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ [CONFIG ERROR] {e}")
    sys.exit(1)

# ── 2. read values from config ────────────────────────────────────────
checkpoint_cfg = cm.get(Checkpoint)
system_cfg = cm.get(System)

SCRIPT = os.path.abspath("mlTrainingModel.py")
CHECKPOINT_DIR = Path(checkpoint_cfg.save_dir)
KEEP_LAST = checkpoint_cfg.keep_last
tracker = MLStateTracker(run_id=system_cfg.run_id, method=checkpoint_cfg.method)

scalar_fields = [
    name for name, value in tracker.__dict__.items()
    if isinstance(value, (int, float))
]
object_fields = [
    name.replace("_state", "") for name in tracker.__dict__
    if name.endswith("_state")
]

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ── 3. simple checkpoint manager using torch.save ─────────────────────

def save_checkpoint(snapshot: dict) -> Path:
    """Saves snapshot atomically. Uses torch.save for tensor support."""
    step = snapshot.get("global_step", 0)
    path = CHECKPOINT_DIR / f"checkpoint_{step:08d}.pt"
    tmp = path.with_suffix(".tmp")
    torch.save(snapshot, tmp)
    os.replace(tmp, path)
    _prune()
    print(f"  [CheckpointManager] saved → {path.name}")
    return path


def load_checkpoint() -> dict | None:
    """Returns the most recent checkpoint or None."""
    files = sorted(CHECKPOINT_DIR.glob("checkpoint_*.pt"))
    if not files:
        return None
    payload = torch.load(files[-1], weights_only=False)
    print(f"  [CheckpointManager] loaded ← {files[-1].name}")
    return payload


def _prune():
    files = sorted(CHECKPOINT_DIR.glob("checkpoint_*.pt"))
    for old in files[:-KEEP_LAST]:
        old.unlink()


# ── 4. build TraceLayer ───────────────────────────────────────────────

layer = TraceLayer(
    target_file=SCRIPT,
    poll_target=scalar_fields,
    snapshot_target=object_fields,
    run_id=system_cfg.run_id,
)


# ── 5. helpers ────────────────────────────────────────────────────────

def _print_state(label: str, state: dict) -> None:
    print(f"\n  [{label}]")
    print(f"    epoch       = {state.get('epoch')}")
    print(f"    batch_idx   = {state.get('batch_idx')}")
    print(f"    global_step = {state.get('global_step')}")
    loss = state.get("loss")
    if loss is not None:
        print(f"    loss        = {loss:.4f}")
    if state.get("model"):
        print(f"    model       = state_dict with {len(state['model'])} keys ✓")
    if state.get("optimizer"):
        print(f"    optimizer   = state_dict saved ✓")
    if state.get("scheduler"):
        print(f"    scheduler   = state_dict saved ✓")


# ── 6. run / resume ───────────────────────────────────────────────────

def run() -> None:
    print("\n" + "=" * 55)
    print(f"  AutoCheck | FRESH START | run_id={system_cfg.run_id}")
    print("  press Ctrl+C at any time to pause")
    print("=" * 55 + "\n")

    layer.attach()
    try:
        runpy.run_path(SCRIPT, run_name="__main__")
        snapshot = layer.snapshot()
        save_checkpoint(snapshot)
        print("\n  [AutoCheck] training complete — final checkpoint saved")

    except KeyboardInterrupt:
        print("\n\n  [AutoCheck] Ctrl+C caught — saving checkpoint...")
        snapshot = layer.snapshot()
        save_checkpoint(snapshot)
        _print_state("AutoCheck saved", snapshot)
        print("\n  run again to resume from this point")

    finally:
        layer.detach()


def resume() -> None:
    print("\n" + "=" * 55)
    print(f"  AutoCheck | RESUME | run_id={system_cfg.run_id}")
    print("=" * 55)

    payload = load_checkpoint()
    if not payload:
        print("  no checkpoint found — starting fresh")
        run()
        return

    _print_state("AutoCheck restoring", payload)
    print(f"\n  [AutoCheck] restoring and resuming...\n")

    layer.attach()
    layer.restore(payload)

    try:
        runpy.run_path(SCRIPT, run_name="__main__")
        snapshot = layer.snapshot()
        save_checkpoint(snapshot)
        print("\n  [AutoCheck] training complete — final checkpoint saved")

    except KeyboardInterrupt:
        print("\n\n  [AutoCheck] paused again — saving checkpoint...")
        snapshot = layer.snapshot()
        save_checkpoint(snapshot)
        _print_state("AutoCheck saved", snapshot)
        print("\n  run again to resume")

    finally:
        layer.detach()


# ── entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    ckpt_files = sorted(CHECKPOINT_DIR.glob("checkpoint_*.pt"))
    if ckpt_files:
        resume()
    else:
        run()
