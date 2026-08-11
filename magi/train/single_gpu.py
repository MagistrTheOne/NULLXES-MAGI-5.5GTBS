"""Production single-GPU MAGI training entry (no T4 / no smoke defaults)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from magi.checkpoint.manifest import file_sha256
from magi.config import ModelConfig, load_model_config, load_simple_yaml
from magi.model import MAGITransformer
from magi.runtime.production import is_smoke_name
from magi.tokenizer import load_tokenizer
from magi.tokenizer.byte_bpe import MagiByteBPETokenizer
from magi.train import TrainConfig, load_train_checkpoint, pack_texts, save_train_checkpoint, train_steps
from magi.train.data import load_corpus_lines, load_shard_batches


def resolve_tokenizer_from_model_config(
    model_config_path: Path,
    *,
    root: Path,
    cfg: ModelConfig,
    allow_runtime_probe: bool = False,
) -> tuple[MagiByteBPETokenizer, Path]:
    raw = load_simple_yaml(model_config_path)
    tok_section = raw.get("tokenizer") or {}
    meta = raw.get("meta") or {}

    if not allow_runtime_probe:
        if meta.get("production_pretraining_allowed") in (False, "false"):
            blocked = meta.get("blocked_until", "MAGI_TOKENIZER_V1")
            raise SystemExit(
                f"REFUSE BASE training: {model_config_path.name} "
                f"production_pretraining_allowed=false (blocked_until={blocked}). "
                "Freeze MAGI_TOKENIZER_V1 first, or pass allow_runtime_probe=True "
                "only for non-BASE engineering checks (no production checkpoint)."
            )

    prod_artifact = tok_section.get("production_artifact")
    bringup_artifact = tok_section.get("bringup_probe_artifact") or tok_section.get("artifact")
    tok_id_hint = str(
        tok_section.get("production_id")
        or tok_section.get("bringup_probe_id")
        or tok_section.get("id")
        or ""
    )

    if allow_runtime_probe:
        artifact = bringup_artifact
        if not artifact:
            raise SystemExit(f"{model_config_path}: no bringup probe tokenizer artifact")
    else:
        artifact = prod_artifact or tok_section.get("artifact")
        if not artifact:
            raise SystemExit(
                f"{model_config_path}: tokenizer.production_artifact required for BASE training"
            )

    path = Path(str(artifact))
    if not path.is_absolute():
        path = (root / path).resolve()
    if not path.exists():
        raise SystemExit(
            f"tokenizer artifact missing: {path}. "
            "BASE requires tokenizer/artifacts/magi_tokenizer_v1.json after freeze."
        )
    if is_smoke_name(path) or is_smoke_name(tok_id_hint):
        raise SystemExit(
            "T4/smoke tokenizer is forbidden. "
            f"id={tok_id_hint!r} artifact={path}"
        )

    # MagiByteBPETokenizer.load for probe; load_tokenizer rejects smoke ids.
    from magi.tokenizer.byte_bpe import MagiByteBPETokenizer

    if allow_runtime_probe:
        tokenizer = MagiByteBPETokenizer.load(path)
    else:
        tokenizer = load_tokenizer(path)
        tid = tokenizer.tokenizer_id.lower()
        if "bringup" in tid or tid.endswith("_8k_v0.1") or "8k" in tid:
            raise SystemExit(
                f"REFUSE: bringup tokenizer {tokenizer.tokenizer_id!r} cannot train MAGI BASE. "
                "Freeze MAGI_TOKENIZER_V1 first."
            )

    if is_smoke_name(tokenizer.tokenizer_id):
        raise SystemExit(
            f"tokenizer_id={tokenizer.tokenizer_id!r} is forbidden in production training"
        )
    if tokenizer.vocab_size != cfg.vocab_size:
        raise SystemExit(
            f"tokenizer vocab {tokenizer.vocab_size} != model vocab {cfg.vocab_size}"
        )
    return tokenizer, path


def run_single_gpu_train(
    *,
    root: Path,
    profile_path: Path,
    device_name: str,
    steps: int | None = None,
    seq_len: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
    checkpoint_dir: Path | None = None,
    checkpoint_every: int | None = None,
    save_optimizer: bool = False,
    resume: Path | None = None,
    skip_checkpoint: bool = False,
    corpus: Path | None = None,
    shard: Path | None = None,
    allow_runtime_probe: bool = False,
) -> int:
    if is_smoke_name(profile_path):
        raise SystemExit(
            f"Refuse T4/smoke train profile in production runner: {profile_path}"
        )

    profile = load_simple_yaml(profile_path)
    if not allow_runtime_probe:
        if profile.get("meta", {}).get("production_pretraining_allowed") in (False, "false"):
            raise SystemExit(
                f"REFUSE: {profile_path.name} production_pretraining_allowed=false. "
                "Current phase is DATA PILOT → MAGI_TOKENIZER_V1. "
                "Pass --allow-runtime-probe only for non-BASE engineering checks."
            )
        if profile.get("train", {}).get("production_checkpoint_allowed") in (False, "false"):
            # Soft signal — still refuse BASE path without frozen tokenizer.
            pass

    train_cfg = profile.get("train", {})
    model_config_path = root / str(profile["meta"]["model_config"])
    if is_smoke_name(model_config_path):
        raise SystemExit(f"Refuse T4/smoke model config: {model_config_path}")

    corpus_path = (
        Path(corpus)
        if corpus is not None
        else root / str(profile["meta"]["corpus"])
    )
    if not corpus_path.is_absolute():
        corpus_path = (root / corpus_path).resolve()

    steps_n = int(steps if steps is not None else train_cfg.get("steps", 20))
    seq_n = int(seq_len if seq_len is not None else train_cfg.get("seq_len", 128))
    batch_n = int(batch_size if batch_size is not None else train_cfg.get("batch_size", 1))
    lr_n = float(lr if lr is not None else train_cfg.get("lr", 1.0e-4))
    require_improve = bool(train_cfg.get("require_loss_improve", False))
    ckpt_dir = checkpoint_dir or root / str(train_cfg.get("checkpoint_dir", "artifacts/magi_train"))
    ckpt_every = int(
        checkpoint_every if checkpoint_every is not None else train_cfg.get("checkpoint_every", 0)
    )
    save_opt = bool(save_optimizer or train_cfg.get("save_optimizer", False))

    import torch

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but not available")

    cfg = load_model_config(model_config_path)
    if is_smoke_name(cfg.name):
        raise SystemExit(f"Refuse T4/smoke model name: {cfg.name}")

    tokenizer, tokenizer_artifact = resolve_tokenizer_from_model_config(
        model_config_path,
        root=root,
        cfg=cfg,
        allow_runtime_probe=allow_runtime_probe,
    )
    if allow_runtime_probe:
        print("WARNING: allow_runtime_probe=1 — NOT a MAGI BASE training run")
        skip_checkpoint = True
        print("WARNING: checkpoints disabled under runtime probe (no disposable BASE ckpt)")


    print("=== MAGI TRAIN ===")
    print(f"model={cfg.name}")
    print(f"model_class={cfg.model_class}")
    print(f"is_moe={cfg.is_moe}")
    if cfg.is_moe:
        print(
            f"moe=routed={cfg.n_routed_experts} shared={cfg.n_shared_experts} "
            f"top_k={cfg.top_k} dense_layers={cfg.n_dense_layers} moe_layers={cfg.n_moe_layers}"
        )
    print(f"tokenizer={tokenizer.tokenizer_id} artifact={tokenizer_artifact}")
    print(f"device={device}")
    print(f"steps={steps_n} seq={seq_n} batch={batch_n} lr={lr_n}")
    print(f"checkpoint_dir={ckpt_dir} every={ckpt_every} save_optimizer={save_opt}")

    model = MAGITransformer.from_config(cfg).to(device=device)

    if shard is not None:
        shard_path = Path(shard)
        if not shard_path.is_absolute():
            shard_path = (root / shard_path).resolve()
        batches = load_shard_batches(
            shard_path,
            batch_size=batch_n,
            device=device,
            pad_id=tokenizer.pad_id,
        )
        print(f"shard={shard_path}")
        print(f"shard_windows_batches={len(batches)}")
    else:
        texts = load_corpus_lines(corpus_path)
        print(f"corpus={corpus_path}")
        print(f"corpus_lines={len(texts)}")
        batches = pack_texts(
            tokenizer,
            texts,
            seq_len=seq_n,
            batch_size=batch_n,
            device=device,
        )
        print(f"packed_batches={len(batches)}")

    tokenizer_sha = file_sha256(tokenizer_artifact)
    data_manifest_path = root / "data" / "program" / "MAGI_DATA_MANIFEST_v0.1.yaml"
    if data_manifest_path.exists():
        dataset_manifest_id = "MAGI_DATA_MANIFEST_v0.1"
        dataset_manifest_sha256 = file_sha256(data_manifest_path)
        mixture_id = "base_mixture_v0_1"
    else:
        dataset_manifest_id = "UNBOUND"
        dataset_manifest_sha256 = "UNBOUND"
        mixture_id = "UNBOUND"

    train_config_payload = {
        "steps": steps_n,
        "seq_len": seq_n,
        "batch_size": batch_n,
        "lr": lr_n,
        "weight_decay": float(train_cfg.get("weight_decay", 0.1)),
        "beta1": float(train_cfg.get("beta1", 0.9)),
        "beta2": float(train_cfg.get("beta2", 0.95)),
        "eps": float(train_cfg.get("eps", 1.0e-8)),
        "max_grad_norm": float(train_cfg.get("max_grad_norm", 1.0)),
        "seed": int(train_cfg.get("seed", 42)),
        "use_amp": bool(train_cfg.get("use_amp", True)) and device.type == "cuda",
        "amp_dtype": str(train_cfg.get("amp_dtype", "bf16")),
        "profile": str(profile_path),
        "model_config": str(model_config_path),
    }

    resume_optimizer = None
    resume_scaler = None
    start_step = 0
    start_tokens = 0
    if resume is not None:
        resume_path = Path(resume)
        if not resume_path.is_absolute():
            resume_path = (root / resume_path).resolve()
        resume_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr_n,
            betas=(train_config_payload["beta1"], train_config_payload["beta2"]),
            eps=train_config_payload["eps"],
            weight_decay=train_config_payload["weight_decay"],
        )
        loaded = load_train_checkpoint(
            resume_path,
            model=model,
            optimizer=resume_optimizer,
            map_location=device,
            restore_rng=True,
        )
        start_step = int(loaded.get("step") or 0)
        start_tokens = int(loaded.get("consumed_tokens") or 0)
        print(f"resume={resume_path} step={start_step} consumed_tokens={start_tokens}")

    result_holder: dict[str, Any] = {"optimizer": resume_optimizer, "scaler": resume_scaler}

    def _write_ckpt(
        step: int,
        loss: float,
        metrics: dict,
        *,
        with_optimizer: bool,
        update_latest: bool,
    ) -> Path:
        if result_holder["optimizer"] is None:
            raise RuntimeError("cannot write checkpoint before optimizer exists")
        return save_train_checkpoint(
            ckpt_dir,
            model=model,
            optimizer=result_holder["optimizer"],
            scaler=result_holder["scaler"],
            step=step,
            loss=loss,
            config_path=model_config_path,
            model_name=cfg.name,
            tokenizer_id=tokenizer.tokenizer_id,
            tokenizer_sha256=tokenizer_sha,
            tokenizer_path=tokenizer_artifact,
            metrics=metrics,
            save_optimizer=with_optimizer,
            update_latest=update_latest,
            model_architecture="MagiForCausalLM",
            model_revision="v0.1",
            dataset_manifest_id=dataset_manifest_id,
            dataset_manifest_sha256=dataset_manifest_sha256,
            mixture_id=mixture_id,
            train_config=train_config_payload,
            consumed_tokens=int(metrics.get("consumed_tokens", start_tokens)),
            consumed_samples=int(step) * int(batch_n),
            compute_dtype=str(train_config_payload["amp_dtype"]),
            save_rng=True,
        )

    def on_step(step, metrics, model_ref, optimizer, scaler):
        result_holder["optimizer"] = optimizer
        result_holder["scaler"] = scaler
        if skip_checkpoint or ckpt_every <= 0:
            return
        if step % ckpt_every != 0:
            return
        path = _write_ckpt(
            step,
            metrics.loss,
            {
                "status": "MID",
                "step": step,
                "loss": metrics.loss,
                "consumed_tokens": start_tokens + (step - start_step) * seq_n * batch_n,
            },
            with_optimizer=save_opt,
            update_latest=False,
        )
        print(f"checkpoint_mid={path}")

    run_steps = steps_n if start_step == 0 else max(1, steps_n - start_step)
    result = train_steps(
        model,
        batches,
        config=TrainConfig(
            steps=run_steps,
            lr=lr_n,
            weight_decay=float(train_cfg.get("weight_decay", 0.1)),
            beta1=float(train_cfg.get("beta1", 0.9)),
            beta2=float(train_cfg.get("beta2", 0.95)),
            eps=float(train_cfg.get("eps", 1.0e-8)),
            max_grad_norm=float(train_cfg.get("max_grad_norm", 1.0)),
            seed=int(train_cfg.get("seed", 42)),
            use_amp=bool(train_cfg.get("use_amp", True)) and device.type == "cuda",
            amp_dtype=str(train_cfg.get("amp_dtype", "bf16")),
            log_every=1,
            checkpoint_every=ckpt_every,
        ),
        on_step=None if skip_checkpoint else on_step,
        optimizer=resume_optimizer,
        scaler=resume_scaler,
        start_step=start_step,
        consumed_tokens=start_tokens,
        reseed=resume is None,
    )
    result_holder["optimizer"] = result.optimizer
    result_holder["scaler"] = result.scaler
    summary = result.summary
    if start_step:
        summary["resumed_from"] = start_step
    print("=== SUMMARY ===")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if not skip_checkpoint and result.history:
        ckpt = _write_ckpt(
            result.history[-1].step,
            result.history[-1].loss,
            summary,
            with_optimizer=save_opt or summary.get("status") in {"OK", "INTERRUPTED"},
            update_latest=True,
        )
        print(f"checkpoint={ckpt}")
        print(f"checkpoint_root={ckpt_dir / 'model.safetensors'}")
        loaded = load_train_checkpoint(
            ckpt_dir / "model.safetensors",
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
