"""Single-GPU training checkpoint I/O — safetensors weights are canonical."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from magi.checkpoint.manifest import build_checkpoint_manifest
from magi.model.torch_runtime import require_torch

torch = require_torch()

CHECKPOINT_FORMAT = "magi_single_gpu_v0.2"


@dataclass(frozen=True)
class TrainCheckpoint:
    step: int
    loss: float
    config_path: str
    model_name: str
    tokenizer_id: str


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
    # Contiguous CPU tensors for portable safetensors dumps.
    state = {k: v.detach().to("cpu").contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(target))
    return target


def load_model_safetensors(model: torch.nn.Module, path: str | Path) -> None:
    _, load_file = _require_safetensors()
    state = load_file(str(path))
    model.load_state_dict(state)


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
) -> Path:
    """Write step dir with model.safetensors (+ optional optimizer.pt) and optional latest pointers."""
    root = Path(output_dir)
    step_dir = root / f"step-{int(step):06d}"
    step_dir.mkdir(parents=True, exist_ok=True)

    weights_path = save_model_safetensors(model, step_dir / "model.safetensors")

    if save_optimizer:
        torch.save(
            {
                "format": CHECKPOINT_FORMAT,
                "step": int(step),
                "optimizer": optimizer.state_dict(),
                "scaler": None if scaler is None else scaler.state_dict(),
            },
            step_dir / "optimizer.pt",
        )

    manifest = build_checkpoint_manifest(
        model_name=model_name,
        config_path=config_path,
        tokenizer_id=tokenizer_id,
        tokenizer_sha256=tokenizer_sha256,
        parallelism={"tp": 1, "pp": 1, "ep": 1, "cp": 1, "dp": 1},
        checkpoint_format=CHECKPOINT_FORMAT,
    )
    meta = {
        "checkpoint": asdict(
            TrainCheckpoint(
                step=step,
                loss=float(loss),
                config_path=str(config_path),
                model_name=model_name,
                tokenizer_id=tokenizer_id,
            )
        ),
        "manifest": manifest.to_dict(),
        "metrics": metrics,
        "weights": "model.safetensors",
        "optimizer": "optimizer.pt" if save_optimizer else None,
    }
    (step_dir / "train_meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    if update_latest:
        save_model_safetensors(model, root / "model.safetensors")
        (root / "train_meta.json").write_text(
            json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return weights_path

def load_train_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    source = Path(path)
    if source.is_dir():
        weights = source / "model.safetensors"
        optim_path = source / "optimizer.pt"
        meta_path = source / "train_meta.json"
    elif source.name == "model.safetensors":
        weights = source
        optim_path = source.parent / "optimizer.pt"
        meta_path = source.parent / "train_meta.json"
    elif source.suffix == ".pt":
        # Legacy v0.1 train.pt
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
        payload["step"] = payload["meta"].get("checkpoint", {}).get("step")

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
    return payload
