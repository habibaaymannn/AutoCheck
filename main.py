"""
main.py
AutoCheck entry point.

Delegates execution to CLI which calls RunnerScript.
"""

from __future__ import annotations

import sys
from runnerscript.cli import main as cli_main


def main() -> int:
    return cli_main()


if __name__ == "__main__":
    sys.exit(main())