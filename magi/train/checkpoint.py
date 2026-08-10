"""Single-GPU training checkpoint I/O for MAGI smokes."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from magi.checkpoint.manifest import build_checkpoint_manifest
from magi.model.torch_runtime import require_torch

torch = require_torch()

CHECKPOINT_FORMAT = "magi_single_gpu_v0.1"


@dataclass(frozen=True)
class TrainCheckpoint:
    step: int
    loss: float
    config_path: str
    model_name: str
    tokenizer_id: str


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
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "step": int(step),
        "loss": float(loss),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": None if scaler is None else scaler.state_dict(),
        "metrics": metrics,
    }
    ckpt_path = output / "train.pt"
    torch.save(payload, ckpt_path)

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
    }
    (output / "train_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return ckpt_path


def load_train_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    try:
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(Path(path), map_location=map_location)
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"unsupported checkpoint format: {payload.get('format')!r}")
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scaler is not None and payload.get("scaler") is not None:
        scaler.load_state_dict(payload["scaler"])
    return payload
