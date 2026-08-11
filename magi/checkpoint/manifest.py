"""Checkpoint manifest contract for MAGI training and serving.

Manifest proves what was actually written — not what might be written someday.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

MANIFEST_VERSION = "1.0"
CHECKPOINT_SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class CheckpointArtifact:
    """One on-disk object that was successfully written for this checkpoint."""

    path: str
    sha256: str
    bytes: int
    kind: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CheckpointManifest:
    # Schema
    manifest_version: str
    checkpoint_format: str
    checkpoint_schema_version: str

    # Model identity
    model_name: str
    model_architecture: str
    model_revision: str
    config_path: str
    config_sha256: str
    total_parameters: int | None
    active_parameters_per_token: int | None

    # Tokenizer
    tokenizer_id: str
    tokenizer_sha256: str

    # Data lineage
    dataset_manifest_id: str
    dataset_manifest_sha256: str
    mixture_id: str

    # Training lineage
    train_config_sha256: str
    run_id: str
    global_step: int
    consumed_tokens: int
    consumed_samples: int | None

    # Distributed / numeric
    world_size: int
    parallelism: dict[str, Any]
    parameter_dtype: str
    compute_dtype: str

    # Actual inventory (writer fills after successful writes)
    artifacts: dict[str, CheckpointArtifact]
    state_sections: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["artifacts"] = {k: v if isinstance(v, dict) else asdict(v) for k, v in self.artifacts.items()}
        # asdict already expands nested dataclasses; ensure state_sections is list for JSON
        payload["state_sections"] = list(self.state_sections)
        return payload


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_from_path(path: str | Path, *, kind: str, relative_to: Path | None = None) -> CheckpointArtifact:
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"checkpoint artifact missing: {target}")
    rel = str(target.relative_to(relative_to)) if relative_to is not None else str(target)
    return CheckpointArtifact(
        path=rel.replace("\\", "/"),
        sha256=file_sha256(target),
        bytes=int(target.stat().st_size),
        kind=kind,
    )


def build_checkpoint_manifest(
    *,
    model_name: str,
    config_path: str | Path,
    tokenizer_id: str,
    tokenizer_sha256: str,
    parallelism: dict[str, Any],
    checkpoint_format: str = "magi_single_gpu_v0.3",
    # Identity
    model_architecture: str = "MagiForCausalLM",
    model_revision: str = "v0.1",
    total_parameters: int | None = None,
    active_parameters_per_token: int | None = None,
    # Data
    dataset_manifest_id: str = "UNBOUND",
    dataset_manifest_sha256: str = "UNBOUND",
    mixture_id: str = "UNBOUND",
    # Train
    train_config_sha256: str = "UNBOUND",
    run_id: str = "UNBOUND",
    global_step: int = 0,
    consumed_tokens: int = 0,
    consumed_samples: int | None = None,
    # Topology / dtype
    world_size: int | None = None,
    parameter_dtype: str = "float32",
    compute_dtype: str = "bfloat16",
    # Inventory — REQUIRED for honest contract; empty only for HF metadata stubs
    artifacts: dict[str, CheckpointArtifact] | None = None,
    manifest_version: str = MANIFEST_VERSION,
    checkpoint_schema_version: str = CHECKPOINT_SCHEMA_VERSION,
    config_sha256: str | None = None,
) -> CheckpointManifest:
    arts = dict(artifacts or {})
    # state_sections derived from what was actually written (+ identity sections always listed)
    written = tuple(sorted(arts.keys()))
    identity_sections = ("config", "tokenizer")
    state_sections = tuple(dict.fromkeys((*written, *identity_sections)))

    para = dict(parallelism)
    ws = int(world_size) if world_size is not None else int(
        para.get("dp", 1) * para.get("tp", 1) * para.get("pp", 1) * para.get("ep", 1) * para.get("cp", 1)
    )

    return CheckpointManifest(
        manifest_version=manifest_version,
        checkpoint_format=checkpoint_format,
        checkpoint_schema_version=checkpoint_schema_version,
        model_name=model_name,
        model_architecture=model_architecture,
        model_revision=model_revision,
        config_path=str(config_path),
        config_sha256=config_sha256 if config_sha256 is not None else file_sha256(config_path),
        total_parameters=total_parameters,
        active_parameters_per_token=active_parameters_per_token,
        tokenizer_id=tokenizer_id,
        tokenizer_sha256=tokenizer_sha256,
        dataset_manifest_id=dataset_manifest_id,
        dataset_manifest_sha256=dataset_manifest_sha256,
        mixture_id=mixture_id,
        train_config_sha256=train_config_sha256,
        run_id=run_id,
        global_step=int(global_step),
        consumed_tokens=int(consumed_tokens),
        consumed_samples=None if consumed_samples is None else int(consumed_samples),
        world_size=ws,
        parallelism=para,
        parameter_dtype=parameter_dtype,
        compute_dtype=compute_dtype,
        artifacts=arts,
        state_sections=state_sections,
    )
