#!/usr/bin/env python3
"""MAGI T4 single-GPU training smoke.

Usage:
  python scripts/t4_train_smoke.py --device cuda
  python scripts/t4_train_smoke.py --device cpu --steps 3 --seq 64
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from magi.checkpoint.manifest import file_sha256
from magi.config import load_model_config, load_simple_yaml
from magi.model import MAGITransformer
from magi.tokenizer import build_t4_smoke_tokenizer
from magi.train import TrainConfig, load_train_checkpoint, pack_texts, save_train_checkpoint, train_steps
from magi.train.data import load_corpus_lines, load_shard_batches


def main() -> int:
    parser = argparse.ArgumentParser(description="MAGI T4 training smoke")
    parser.add_argument(
        "--profile",
        type=Path,
        default=ROOT / "configs" / "magi_t4_train_smoke_v0.1.yaml",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--seq", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--skip-checkpoint", action="store_true")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=None,
        help="plain .txt or synthetic records.jsonl (overrides profile corpus)",
    )
    parser.add_argument(
        "--shard",
        type=Path,
        default=None,
        help="packed shard .bin from build_synthetic_dataset.py",
    )
    args = parser.parse_args()

    profile = load_simple_yaml(args.profile)
    train_cfg = profile.get("train", {})
    model_config_path = ROOT / str(profile.get("meta", {}).get("model_config", "configs/magi_t4_smoke_v0.1.yaml"))
    corpus_path = Path(args.corpus) if args.corpus is not None else ROOT / str(
        profile.get("meta", {}).get("corpus", "tokenizer/data/t4_smoke_seed.txt")
    )
    if not corpus_path.is_absolute():
        corpus_path = (ROOT / corpus_path).resolve()

    steps = int(args.steps if args.steps is not None else train_cfg.get("steps", 20))
    seq_len = int(args.seq if args.seq is not None else train_cfg.get("seq_len", 128))
    batch_size = int(args.batch_size if args.batch_size is not None else train_cfg.get("batch_size", 1))
    lr = float(args.lr if args.lr is not None else train_cfg.get("lr", 1.0e-4))
    require_improve = bool(train_cfg.get("require_loss_improve", True))
    checkpoint_dir = args.checkpoint_dir or ROOT / str(train_cfg.get("checkpoint_dir", "artifacts/t4_train_smoke"))

    import torch

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but not available")

    cfg = load_model_config(model_config_path)
    tokenizer = build_t4_smoke_tokenizer(vocab_size=cfg.vocab_size)

    print("=== TRAIN SMOKE ===")
    print(f"model={cfg.name}")
    print(f"device={device}")
    print(f"steps={steps} seq={seq_len} batch={batch_size} lr={lr}")

    # Keep master weights in fp32. CUDA AMP autocast handles compute dtype;
    # GradScaler rejects FP16 parameter gradients.
    model = MAGITransformer.from_config(cfg).to(device=device)

    if args.shard is not None:
        shard_path = Path(args.shard)
        if not shard_path.is_absolute():
            shard_path = (ROOT / shard_path).resolve()
        batches = load_shard_batches(
            shard_path,
            batch_size=batch_size,
            device=device,
            pad_id=tokenizer.pad_id,
        )
        print(f"shard={shard_path}")
        print(f"shard_windows_batches={len(batches)}")
        corpus_hash_src = shard_path
    else:
        texts = load_corpus_lines(corpus_path)
        print(f"corpus={corpus_path}")
        print(f"corpus_lines={len(texts)}")
        batches = pack_texts(
            tokenizer,
            texts,
            seq_len=seq_len,
            batch_size=batch_size,
            device=device,
        )
        print(f"packed_batches={len(batches)}")
        corpus_hash_src = corpus_path

    result = train_steps(
        model,
        batches,
        config=TrainConfig(
            steps=steps,
            lr=lr,
            weight_decay=float(train_cfg.get("weight_decay", 0.1)),
            beta1=float(train_cfg.get("beta1", 0.9)),
            beta2=float(train_cfg.get("beta2", 0.95)),
            eps=float(train_cfg.get("eps", 1.0e-8)),
            max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
            seed=int(train_cfg.get("seed", 42)),
            use_amp=bool(train_cfg.get("use_amp", True)) and device.type == "cuda",
            amp_dtype=str(train_cfg.get("amp_dtype", "fp16")),
            log_every=1,
        ),
    )
    summary = result.summary
    print("=== SUMMARY ===")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if summary["status"] == "NAN_STOP":
        raise SystemExit("training stopped on NaN")
    if require_improve and not summary.get("loss_improved", False):
        raise SystemExit(
            f"loss did not improve: first={summary['first_loss']} last={summary['last_loss']}"
        )

    if not args.skip_checkpoint:
        ckpt = save_train_checkpoint(
            checkpoint_dir,
            model=model,
            optimizer=result.optimizer,
            scaler=result.scaler,
            step=result.history[-1].step,
            loss=result.history[-1].loss,
            config_path=model_config_path,
            model_name=cfg.name,
            tokenizer_id=tokenizer.tokenizer_id,
            tokenizer_sha256=file_sha256(corpus_hash_src),
            metrics=summary,
        )
        loaded = load_train_checkpoint(ckpt, model=MAGITransformer.from_config(cfg), map_location="cpu")
        print(f"checkpoint={ckpt}")
        print(f"checkpoint_reload_step={loaded['step']}")

    if device.type == "cuda":
        print(f"cuda_allocated_gb={torch.cuda.memory_allocated(device)/(1024**3):.3f}")
        print(f"cuda_reserved_gb={torch.cuda.memory_reserved(device)/(1024**3):.3f}")
    print("status=OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
