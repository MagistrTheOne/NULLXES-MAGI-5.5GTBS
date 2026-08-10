#!/usr/bin/env python3
"""Build MAGI synthetic dataset (jsonl + manifests + packed shards).

Usage:
  python scripts/build_synthetic_dataset.py
  python scripts/build_synthetic_dataset.py --docs 5000 --seed 42 --seq 128
  python scripts/build_synthetic_dataset.py --golden
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from magi.config import load_simple_yaml
from magi.data.synthetic import build_synthetic_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Build MAGI synthetic dataset")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "synthetic_magi_v0.1.yaml")
    parser.add_argument("--docs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seq", type=int, default=None)
    parser.add_argument("--tokenizer-vocab", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--golden", action="store_true", help="Write small golden sample into data/")
    parser.add_argument("--no-shards", action="store_true")
    args = parser.parse_args()

    cfg = load_simple_yaml(args.config)
    meta = cfg.get("meta", {})
    gen = cfg.get("generate", {})
    tok = cfg.get("tokenizer", {})
    packing = cfg.get("packing", {})
    out_cfg = cfg.get("output", {})
    weights = {str(k): float(v) for k, v in dict(cfg.get("domain_weights", {})).items()}

    if args.golden:
        n_docs = int(out_cfg.get("golden_docs", 64))
        output = ROOT / str(out_cfg.get("golden_root", "data/synthetic/magi_synth_v0.1/golden"))
        seed = int(args.seed if args.seed is not None else gen.get("seed", 42))
    else:
        n_docs = int(args.docs if args.docs is not None else gen.get("n_docs", 5000))
        output = args.output or ROOT / str(out_cfg.get("root", "artifacts/synthetic/magi_synth_v0.1"))
        seed = int(args.seed if args.seed is not None else gen.get("seed", 42))

    seq_len = int(args.seq if args.seq is not None else packing.get("seq_len", 128))
    vocab = int(args.tokenizer_vocab if args.tokenizer_vocab is not None else tok.get("vocab_size", 8192))

    report = build_synthetic_dataset(
        output_dir=output,
        n_docs=n_docs,
        seed=seed,
        seq_len=seq_len,
        tokenizer_vocab=vocab,
        domain_weights=weights,
        dataset_id=str(meta.get("dataset_id", "magi_synth_v0.1")),
        version=str(meta.get("version", "v0.1")),
        write_shards=not args.no_shards,
        config_path=args.config,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"output={output}")
    print("status=OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
