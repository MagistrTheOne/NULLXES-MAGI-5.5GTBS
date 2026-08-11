"""Serialization helpers for MAGI HF checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from magi.checkpoint import CheckpointManifest, build_checkpoint_manifest
from magi.hf.configuration_magi import MagiConfig
from magi.hf.convert import hf_state_dict_to_native, native_state_dict_to_hf
from magi.hf.generation import build_generation_config
from magi.hf.versions import SERIALIZATION_VERSION


def safetensors_available() -> bool:
    try:
        import safetensors  # noqa: F401
    except ImportError:
        return False
    return True


def save_config_bundle(config: MagiConfig, output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if config.serialization_version != SERIALIZATION_VERSION:
        raise ValueError(
            f"Unsupported serialization_version={config.serialization_version!r}; "
            f"expected {SERIALIZATION_VERSION!r}"
        )
    config.save_pretrained(output)
    build_generation_config(config).save_pretrained(output)


def save_state_dict(
    state_dict: Mapping[str, torch.Tensor],
    output_dir: str | Path,
    *,
    native: bool = False,
) -> Path:
    """Save weights as safetensors when available, otherwise pytorch_model.bin."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    payload = dict(state_dict) if not native else native_state_dict_to_hf(state_dict)
    if safetensors_available():
        from safetensors.torch import save_file

        path = output / "model.safetensors"
        save_file(payload, str(path))
        return path
    path = output / "pytorch_model.bin"
    torch.save(payload, path)
    return path


def load_state_dict(path: str | Path, *, to_native: bool = False) -> dict[str, torch.Tensor]:
    source = Path(path)
    if source.is_dir():
        safetensors_path = source / "model.safetensors"
        bin_path = source / "pytorch_model.bin"
        if safetensors_path.exists():
            source = safetensors_path
        elif bin_path.exists():
            source = bin_path
        else:
            raise FileNotFoundError(f"No model weights found under {path}")
    if source.suffix == ".safetensors":
        if not safetensors_available():
            raise RuntimeError("safetensors is required to load model.safetensors")
        from safetensors.torch import load_file

        state = load_file(str(source))
    else:
        state = torch.load(source, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise TypeError(f"Unexpected checkpoint payload type: {type(state)!r}")
    return hf_state_dict_to_native(state) if to_native else dict(state)


def build_hf_checkpoint_manifest(
    *,
    config: MagiConfig,
    config_path: str | Path,
    tokenizer_id: str,
    tokenizer_sha256: str,
    parallelism: dict[str, Any],
    artifacts: dict[str, Any] | None = None,
    global_step: int = 0,
    consumed_tokens: int = 0,
    train_config_sha256: str = "UNBOUND",
    run_id: str = "UNBOUND",
    dataset_manifest_id: str = "UNBOUND",
    dataset_manifest_sha256: str = "UNBOUND",
    mixture_id: str = "UNBOUND",
) -> CheckpointManifest:
    return build_checkpoint_manifest(
        model_name=config.name,
        config_path=config_path,
        tokenizer_id=tokenizer_id,
        tokenizer_sha256=tokenizer_sha256,
        parallelism=parallelism,
        checkpoint_format="hf_pretrained",
        model_architecture="MagiForCausalLM",
        model_revision=getattr(config, "config_version", "v0.1"),
        artifacts=artifacts or {},
        global_step=global_step,
        consumed_tokens=consumed_tokens,
        train_config_sha256=train_config_sha256,
        run_id=run_id,
        dataset_manifest_id=dataset_manifest_id,
        dataset_manifest_sha256=dataset_manifest_sha256,
        mixture_id=mixture_id,
    )
