#!/usr/bin/env python3
"""Validate MAGI model config and print core fields."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from magi.config import load_model_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate MAGI model config")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    cfg = load_model_config(args.config)
    print(f"name={cfg.name}")
    print(f"model_class={cfg.model_class}")
    print(f"d_model={cfg.d_model}")
    print(f"n_layers={cfg.n_layers}")
    print(f"n_heads={cfg.n_heads}")
    print(f"n_kv_heads={cfg.n_kv_heads}")
    print(f"vocab_size={cfg.vocab_size}")
    print(f"is_moe={cfg.is_moe}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
