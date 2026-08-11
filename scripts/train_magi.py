#!/usr/bin/env python3
"""Canonical MAGI Sparse-MoE training entry (from-zero weights).

No T4. No smoke defaults. Tokenizer must come from model config artifact.

Usage:
  python scripts/train_magi.py --device cuda
  python scripts/train_magi.py --device cuda --steps 100 --seq 1024
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from magi.train.single_gpu import run_single_gpu_train


def main() -> int:
    parser = argparse.ArgumentParser(description="MAGI production single-GPU train")
    parser.add_argument(
        "--profile",
        type=Path,
        default=ROOT / "configs" / "magi_7b_train_h200_v0.1.yaml",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seq", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--save-optimizer", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--skip-checkpoint", action="store_true")
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--shard", type=Path, default=None)
    parser.add_argument(
        "--allow-runtime-probe",
        action="store_true",
        help="non-BASE engineering probe only; forbids production checkpoints; may use bringup_8k",
    )
    parser.add_argument(
        "--save-probe-checkpoint",
        action="store_true",
        help="with --allow-runtime-probe: write PROBE_NOT_BASE weights (still not BASE)",
    )
    args = parser.parse_args()

    return run_single_gpu_train(
        root=ROOT,
        profile_path=args.profile if args.profile.is_absolute() else (ROOT / args.profile),
        device_name=args.device,
        steps=args.steps,
        seq_len=args.seq,
        batch_size=args.batch_size,
        lr=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        save_optimizer=args.save_optimizer,
        resume=args.resume,
        skip_checkpoint=args.skip_checkpoint,
        corpus=args.corpus,
        shard=args.shard,
        allow_runtime_probe=args.allow_runtime_probe,
        save_probe_checkpoint=args.save_probe_checkpoint,
    )


if __name__ == "__main__":
    raise SystemExit(main())
