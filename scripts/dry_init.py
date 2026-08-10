#!/usr/bin/env python3
"""Dry initialize a MAGI model on a selected device."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from magi.runtime.dry_init import dry_init_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Dry initialize MAGI model")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--device", default="meta", help="Use meta for no memory allocation")
    args = parser.parse_args()

    model, count = dry_init_model(args.config, args.device)
    print(f"name={model.cfg.name}")
    print(f"device={args.device}")
    print(f"parameters={count}")
    print(f"trainable_parameters={model.trainable_parameter_count()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
