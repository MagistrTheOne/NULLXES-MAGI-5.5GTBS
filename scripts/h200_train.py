#!/usr/bin/env python3
"""MAGI H200 intelligence bring-up trainer (MAGI-7B default).

Wrapper around the shared single-GPU train path with H200/7B defaults.

Usage:
  python scripts/h200_train.py --device cuda
  python scripts/h200_train.py --device cuda --steps 500 --seq 1024
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import t4_train_smoke


def main() -> int:
    # Inject H200 profile as default when user did not pass --profile.
    if "--profile" not in sys.argv:
        sys.argv.extend(["--profile", str(ROOT / "configs" / "magi_7b_train_h200_v0.1.yaml")])
    if "--device" not in sys.argv:
        sys.argv.extend(["--device", "cuda"])
    return t4_train_smoke.main()


if __name__ == "__main__":
    raise SystemExit(main())
