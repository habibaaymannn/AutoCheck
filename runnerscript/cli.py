# cli.py
"""
Usage
-----
  autocheck run    --config config.yaml userprogram.py
  autocheck stop   --config config.yaml userprogram.py
  autocheck resume --config config.yaml userprogram.py

  # with optional overrides
  autocheck run    --config config.yaml --mode ml --save-dir ./ckpts userprogram.py
  autocheck run    --config config.yaml --validate-only userprogram.py
  autocheck resume --config config.yaml --checkpoint ./ckpts/ckpt_42.pt userprogram.py
"""

from __future__ import annotations
import argparse
import sys
from typing import Optional
from config.ConfigManager import ConfigParseError, ConfigValidationError
from runnerscript.RunnerScript import RunnerScript

# =============================================================================
# Parser
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autocheck",
        description="Automated checkpointing for ML and HPC jobs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  autocheck run    --config config.yaml train.py
  autocheck run    --config config.yaml --mode ml --save-dir ./ckpts train.py
  autocheck run    --config config.yaml --validate-only train.py
  autocheck stop   --config config.yaml train.py
  autocheck resume --config config.yaml train.py
  autocheck resume --config config.yaml --checkpoint ./ckpts/ckpt_10.pt train.py
        """,
    )

    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    subparsers.required = True  # print help if no subcommand given

    # ── shared arguments reused across all subcommands ────────────────────────
    def _add_shared(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--config", "-c",
            required=True,
            metavar="PATH",
            help="Path to the YAML config file",
        )
        p.add_argument(
            "user_program",
            metavar="userprogram.py",
            help="Path to the user script to run / stop / resume",
        )
        p.add_argument(
            "--mode",
            choices=["ml", "hpc"],
            default=None,
            help="Override execution_mode from config  (ml | hpc)",
        )
        p.add_argument(
            "--save-dir", "-s",
            default=None,
            dest="save_dir",
            metavar="DIR",
            help="Override checkpoint.save_dir from config",
        )

    # ── autocheck run ─────────────────────────────────────────────────────────
    run_p = subparsers.add_parser(
        "run",
        help="Start a fresh run with checkpointing enabled",
    )
    _add_shared(run_p)
    run_p.add_argument(
        "--validate-only",
        action="store_true",
        default=False,
        dest="validate_only",
        help="Validate config and exit without launching the user program",
    )

    # ── autocheck stop ────────────────────────────────────────────────────────
    stop_p = subparsers.add_parser(
        "stop",
        help="Gracefully stop a running job and save a final checkpoint",
    )
    _add_shared(stop_p)

    # ── autocheck resume ──────────────────────────────────────────────────────
    resume_p = subparsers.add_parser(
        "resume",
        help="Resume from the latest checkpoint (or a specific one)",
    )
    _add_shared(resume_p)
    resume_p.add_argument(
        "--checkpoint",
        default=None,
        metavar="PATH",
        dest="checkpoint_path",
        help="Resume from a specific checkpoint file (default: latest)",
    )

    return parser


# =============================================================================
# Dispatch
# =============================================================================

def main(argv: Optional[list[str]] = None) -> int:
    """
    Entry point — parse argv and delegate to RunnerScript.

    Returns
    -------
    int  exit code:  0 = success | 1 = user / config error | 2 = unexpected error
    """
    parser = _build_parser()
    args   = parser.parse_args(argv)          # argv=None → reads sys.argv[1:]

    runner = RunnerScript()

    try:
        if args.command == "run":
            runner.run(
                config_path      = args.config,
                user_program     = args.user_program,
                mode_override    = args.mode,
                save_dir_override= args.save_dir,
                validate_only    = args.validate_only,
            )

        elif args.command == "stop":
            runner.stop(
                config_path      = args.config,
                user_program     = args.user_program,
                mode_override    = args.mode,
                save_dir_override= args.save_dir,
            )

        elif args.command == "resume":
           runner.resume(
                config_path      = args.config,
                user_program     = args.user_program,
                mode_override    = args.mode,
             save_dir_override= args.save_dir,)

    except ConfigParseError as exc:
        print(f"\n[autocheck] config parse error:\n  {exc}\n", file=sys.stderr)
        return 1

    except ConfigValidationError as exc:
        print(f"\n[autocheck] config validation error:\n  {exc}\n", file=sys.stderr)
        return 1

    except ValueError as exc:
        print(f"\n[autocheck] error:\n  {exc}\n", file=sys.stderr)
        return 1

    except Exception as exc:
        print(f"\n[autocheck] unexpected error:\n  {exc}\n", file=sys.stderr)
        return 2

    return 0


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    sys.exit(main())