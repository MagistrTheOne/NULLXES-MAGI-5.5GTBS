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
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=None,
        help="write model.safetensors every N steps (0=final only)",
    )
    parser.add_argument(
        "--save-optimizer",
        action="store_true",
        help="also dump optimizer.pt (~8 bytes/param) with mid/final checkpoints",
    )
    parser.add_argument("--resume", type=Path, default=None, help="step dir or model.safetensors")
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
    checkpoint_every = int(
        args.checkpoint_every
        if args.checkpoint_every is not None
        else train_cfg.get("checkpoint_every", 0)
    )
    save_optimizer = bool(args.save_optimizer or train_cfg.get("save_optimizer", False))

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
    print(f"checkpoint_dir={checkpoint_dir} every={checkpoint_every} save_optimizer={save_optimizer}")

    # Keep master weights in fp32. CUDA AMP autocast handles compute dtype.
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

    tokenizer_sha = file_sha256(corpus_hash_src)
    start_step = 0
    if args.resume is not None:
        resume_path = Path(args.resume)
        if not resume_path.is_absolute():
            resume_path = (ROOT / resume_path).resolve()
        loaded = load_train_checkpoint(resume_path, model=model, map_location=device)
        start_step = int(loaded.get("step") or 0)
        print(f"resume={resume_path} step={start_step}")

    def _write_ckpt(
        step: int,
        loss: float,
        metrics: dict,
        *,
        with_optimizer: bool,
        update_latest: bool,
    ) -> Path:
        return save_train_checkpoint(
            checkpoint_dir,
            model=model,
            optimizer=result_holder["optimizer"],
            scaler=result_holder["scaler"],
            step=step,
            loss=loss,
            config_path=model_config_path,
            model_name=cfg.name,
            tokenizer_id=tokenizer.tokenizer_id,
            tokenizer_sha256=tokenizer_sha,
            metrics=metrics,
            save_optimizer=with_optimizer,
            update_latest=update_latest,
        )

    result_holder: dict = {"optimizer": None, "scaler": None}

    def on_step(step, metrics, model_ref, optimizer, scaler):
        result_holder["optimizer"] = optimizer
        result_holder["scaler"] = scaler
        if args.skip_checkpoint or checkpoint_every <= 0:
            return
        if step % checkpoint_every != 0:
            return
        path = _write_ckpt(
            step,
            metrics.loss,
            {"status": "MID", "step": step, "loss": metrics.loss},
            with_optimizer=save_optimizer,
            update_latest=False,
        )
        print(f"checkpoint_mid={path}")

    run_steps = steps if start_step == 0 else max(1, steps - start_step)

    result = train_steps(
        model,
        batches,
        config=TrainConfig(
            steps=run_steps,
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
            checkpoint_every=checkpoint_every,
        ),
        on_step=None if args.skip_checkpoint else on_step,
    )
    result_holder["optimizer"] = result.optimizer
    result_holder["scaler"] = result.scaler
    summary = result.summary
    # Renumber absolute steps if resumed.
    if start_step:
        for item in result.history:
            item.step += start_step
        summary["steps"] = result.history[-1].step
        summary["resumed_from"] = start_step

    print("=== SUMMARY ===")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if not args.skip_checkpoint and result.history:
        ckpt = _write_ckpt(
            result.history[-1].step,
            result.history[-1].loss,
            summary,
            with_optimizer=save_optimizer or summary.get("status") in {"OK", "INTERRUPTED"},
            update_latest=True,
        )
        print(f"checkpoint={ckpt}")
        print(f"checkpoint_root={checkpoint_dir / 'model.safetensors'}")
        loaded = load_train_checkpoint(
            checkpoint_dir / "model.safetensors",
            model=MAGITransformer.from_config(cfg),
            map_location="cpu",
        )
        print(f"checkpoint_reload_step={loaded.get('step')}")

    if summary["status"] == "NAN_STOP":
        raise SystemExit("training stopped on NaN")
    if summary["status"] == "INTERRUPTED":
        print("status=INTERRUPTED")
        return 130
    if require_improve and not summary.get("loss_improved", False):
        raise SystemExit(
            f"loss did not improve: first={summary['first_loss']} last={summary['last_loss']}"
        )

    if device.type == "cuda":
        print(f"cuda_allocated_gb={torch.cuda.memory_allocated(device)/(1024**3):.3f}")
        print(f"cuda_reserved_gb={torch.cuda.memory_reserved(device)/(1024**3):.3f}")
    print("status=OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
