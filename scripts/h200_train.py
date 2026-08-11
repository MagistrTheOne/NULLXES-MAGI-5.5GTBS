#!/usr/bin/env python3
"""MAGI H200 Scale-0 train alias for train_magi.py.

BASE pretraining requires MAGI_TOKENIZER_V1.
bringup_8k only via --allow-runtime-probe (no production checkpoint).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train_magi


def main() -> int:
    return train_magi.main()


if __name__ == "__main__":
    raise SystemExit(main())
