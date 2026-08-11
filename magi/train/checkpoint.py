"""Single-GPU training checkpoint I/O — safetensors weights are canonical."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

from magi.checkpoint.manifest import artifact_from_path, build_checkpoint_manifest
from magi.model.torch_runtime import require_torch

torch = require_torch()

CHECKPOINT_FORMAT = "magi_single_gpu_v0.3"


@dataclass(frozen=True)
class TrainCheckpoint:
    step: int
    loss: float
    config_path: str
    model_name: str
    tokenizer_id: str
    global_step: int
    consumed_tokens: int


def _require_safetensors():
    try:
        from safetensors.torch import load_file, save_file
    except ImportError as exc:
        raise ImportError(
            "safetensors is required for MAGI training checkpoints. "
            "pip install safetensors"
        ) from exc
    return save_file, load_file


def save_model_safetensors(model: torch.nn.Module, path: str | Path) -> Path:
    save_file, _ = _require_safetensors()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().to("cpu").contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(target))
    return target


def load_model_safetensors(model: torch.nn.Module, path: str | Path) -> None:
    _, load_file = _require_safetensors()
    state = load_file(str(path))
    model.load_state_dict(state)


def _stable_json_sha256(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return sha256(blob).hexdigest()


def capture_rng_state() -> dict[str, Any]:
    """Capture RNG bags needed for resume (CPU torch + all CUDA devices + python)."""
    import random

    payload: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda"] = torch.cuda.get_rng_state_all()
    return payload


def restore_rng_state(payload: dict[str, Any]) -> None:
    import random

    if "python" in payload:
        random.setstate(payload["python"])
    if "torch" in payload:
        torch.set_rng_state(payload["torch"])
    if "cuda" in payload and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["cuda"])


def save_train_checkpoint(
    output_dir: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any | None,
    step: int,
    loss: float,
    config_path: str | Path,
    model_name: str,
    tokenizer_id: str,
    tokenizer_sha256: str,
    metrics: dict[str, Any],
    save_optimizer: bool = True,
    update_latest: bool = True,
    tokenizer_path: str | Path | None = None,
    # Lineage / progress
    model_architecture: str = "MagiForCausalLM",
    model_revision: str = "v0.1",
    total_parameters: int | None = None,
    active_parameters_per_token: int | None = None,
    dataset_manifest_id: str = "UNBOUND",
    dataset_manifest_sha256: str = "UNBOUND",
    mixture_id: str = "UNBOUND",
    train_config: dict[str, Any] | None = None,
    train_config_sha256: str | None = None,
    run_id: str | None = None,
    consumed_tokens: int | None = None,
    consumed_samples: int | None = None,
    parallelism: dict[str, Any] | None = None,
    parameter_dtype: str = "float32",
    compute_dtype: str = "bfloat16",
    save_rng: bool = True,
) -> Path:
    """Write step dir with model.safetensors (+ optional optimizer/rng) and inventory manifest."""
    root = Path(output_dir)
    step_dir = root / f"step-{int(step):06d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    if total_parameters is None:
        total_parameters = int(sum(p.numel() for p in model.parameters()))

    para = parallelism or {"tp": 1, "pp": 1, "ep": 1, "cp": 1, "dp": 1}
    run = run_id or str(uuid.uuid4())
    train_payload = train_config or {}
    train_sha = train_config_sha256 or (
        _stable_json_sha256(train_payload) if train_payload else "UNBOUND"
    )
    tokens = int(consumed_tokens) if consumed_tokens is not None else int(metrics.get("consumed_tokens", 0))

    weights_path = save_model_safetensors(model, step_dir / "model.safetensors")
    artifacts = {
        "model": artifact_from_path(weights_path, kind="model", relative_to=step_dir),
    }

    if save_optimizer:
        optim_path = step_dir / "optimizer.pt"
        torch.save(
            {
                "format": CHECKPOINT_FORMAT,
                "step": int(step),
                "optimizer": optimizer.state_dict(),
                "scaler": None if scaler is None else scaler.state_dict(),
            },
            optim_path,
        )
        artifacts["optimizer"] = artifact_from_path(optim_path, kind="optimizer", relative_to=step_dir)

    if save_rng:
        rng_path = step_dir / "rng.pt"
        torch.save({"format": CHECKPOINT_FORMAT, "rng": capture_rng_state()}, rng_path)
        artifacts["rng"] = artifact_from_path(rng_path, kind="rng", relative_to=step_dir)

    manifest = build_checkpoint_manifest(
        model_name=model_name,
        config_path=config_path,
        tokenizer_id=tokenizer_id,
        tokenizer_sha256=tokenizer_sha256,
        parallelism=para,
        checkpoint_format=CHECKPOINT_FORMAT,
        model_architecture=model_architecture,
        model_revision=model_revision,
        total_parameters=total_parameters,
        active_parameters_per_token=active_parameters_per_token,
        dataset_manifest_id=dataset_manifest_id,
        dataset_manifest_sha256=dataset_manifest_sha256,
        mixture_id=mixture_id,
        train_config_sha256=train_sha,
        run_id=run,
        global_step=int(step),
        consumed_tokens=tokens,
        consumed_samples=consumed_samples,
        parameter_dtype=parameter_dtype,
        compute_dtype=compute_dtype,
        artifacts=artifacts,
    )

    tok_path_str = None if tokenizer_path is None else str(tokenizer_path)
    # train_meta.json embeds the manifest; it is NOT listed in artifacts (self-hash cycle).
    meta = {
        "checkpoint": asdict(
            TrainCheckpoint(
                step=step,
                loss=float(loss),
                config_path=str(config_path),
                model_name=model_name,
                tokenizer_id=tokenizer_id,
                global_step=int(step),
                consumed_tokens=tokens,
            )
        ),
        "manifest": manifest.to_dict(),
        "metrics": metrics,
        "train_config": train_payload,
        "weights": "model.safetensors",
        "optimizer": "optimizer.pt" if save_optimizer else None,
        "rng": "rng.pt" if save_rng else None,
        "tokenizer_id": tokenizer_id,
        "tokenizer_path": tok_path_str,
        "tokenizer_artifact": tok_path_str,
    }
    (step_dir / "train_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if update_latest:
        save_model_safetensors(model, root / "model.safetensors")
        (root / "train_meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if save_optimizer and (step_dir / "optimizer.pt").exists():
            import shutil

            shutil.copy2(step_dir / "optimizer.pt", root / "optimizer.pt")
        if save_rng and (step_dir / "rng.pt").exists():
            import shutil

            shutil.copy2(step_dir / "rng.pt", root / "rng.pt")
    return weights_path


def load_train_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = False,
) -> dict[str, Any]:
    source = Path(path)
    if source.is_dir():
        weights = source / "model.safetensors"
        optim_path = source / "optimizer.pt"
        meta_path = source / "train_meta.json"
        rng_path = source / "rng.pt"
    elif source.name == "model.safetensors":
        weights = source
        optim_path = source.parent / "optimizer.pt"
        meta_path = source.parent / "train_meta.json"
        rng_path = source.parent / "rng.pt"
    elif source.suffix == ".pt":
        try:
            payload = torch.load(source, map_location=map_location, weights_only=False)
        except TypeError:
            payload = torch.load(source, map_location=map_location)
        model.load_state_dict(payload["model"])
        if optimizer is not None and payload.get("optimizer") is not None:
            optimizer.load_state_dict(payload["optimizer"])
        if scaler is not None and payload.get("scaler") is not None:
            scaler.load_state_dict(payload["scaler"])
        return payload
    else:
        raise FileNotFoundError(f"unsupported checkpoint path: {path}")

    if not weights.exists():
        raise FileNotFoundError(f"missing model.safetensors under {source}")
    load_model_safetensors(model, weights)

    payload: dict[str, Any] = {"format": CHECKPOINT_FORMAT, "weights": str(weights)}
    if meta_path.exists():
        payload["meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
        ckpt = payload["meta"].get("checkpoint", {})
        payload["step"] = ckpt.get("global_step", ckpt.get("step"))
        payload["consumed_tokens"] = ckpt.get("consumed_tokens")
        payload["manifest"] = payload["meta"].get("manifest")

    if optimizer is not None and optim_path.exists():
        try:
            optim_payload = torch.load(optim_path, map_location=map_location, weights_only=False)
        except TypeError:
            optim_payload = torch.load(optim_path, map_location=map_location)
        if optim_payload.get("optimizer") is not None:
            optimizer.load_state_dict(optim_payload["optimizer"])
        if scaler is not None and optim_payload.get("scaler") is not None:
            scaler.load_state_dict(optim_payload["scaler"])
        payload["step"] = optim_payload.get("step", payload.get("step"))

    if restore_rng and rng_path.exists():
        try:
            rng_payload = torch.load(rng_path, map_location="cpu", weights_only=False)
        except TypeError:
            rng_payload = torch.load(rng_path, map_location="cpu")
        if isinstance(rng_payload, dict) and "rng" in rng_payload:
            restore_rng_state(rng_payload["rng"])
            payload["rng_restored"] = True

    return payload
