"""Checkpoint manifest contract for MAGI training and serving."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CheckpointManifest:
    model_name: str
    config_path: str
    config_sha256: str
    tokenizer_id: str
    tokenizer_sha256: str
    checkpoint_format: str
    parallelism: dict[str, Any]
    state_sections: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_checkpoint_manifest(
    *,
    model_name: str,
    config_path: str | Path,
    tokenizer_id: str,
    tokenizer_sha256: str,
    parallelism: dict[str, Any],
    checkpoint_format: str = "torch_dist",
) -> CheckpointManifest:
    return CheckpointManifest(
        model_name=model_name,
        config_path=str(config_path),
        config_sha256=file_sha256(config_path),
        tokenizer_id=tokenizer_id,
        tokenizer_sha256=tokenizer_sha256,
        checkpoint_format=checkpoint_format,
        parallelism=dict(parallelism),
        state_sections=(
            "model",
            "optimizer",
            "scheduler",
            "rng",
            "data_iterator",
            "config",
            "tokenizer",
        ),
    )
