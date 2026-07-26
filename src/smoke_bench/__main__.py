"""Entry point: ``python -m smoke_bench`` or installed ``smokebench`` script."""

from __future__ import annotations

import sys


def main() -> int:
    from smoke_bench.app import run

    try:
        return run()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
