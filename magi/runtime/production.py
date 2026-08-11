"""Production inference guards for MAGI runtime entrypoints.

Smoke / tiny / fixture tokenizers and configs are forbidden here.
Train and scripts/dev may still use smoke explicitly.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from magi.config import load_model_config, load_simple_yaml
from magi.tokenizer.byte_bpe import MagiByteBPETokenizer, load_tokenizer

_SMOKE_RE = re.compile(
    r"(smoke|tiny|fixture|mock|toy|debug_model|test_model|t4_smoke|magi_t4|magi-t4)",
    re.IGNORECASE,
)


def is_smoke_name(value: str | Path | None) -> bool:
    if value is None:
        return False
    return bool(_SMOKE_RE.search(str(value)))


def assert_production_config(config_path: Path, *, model_name: str) -> None:
    if is_smoke_name(config_path) or is_smoke_name(model_name):
        raise SystemExit(
            "Smoke / tiny / fixture configs are forbidden in production MAGI inference. "
            f"config={config_path} model={model_name}. Use scripts/dev/ for smoke gates."
        )


def assert_production_tokenizer(tokenizer: MagiByteBPETokenizer, *, path: Path | None = None) -> None:
    if is_smoke_name(tokenizer.tokenizer_id) or is_smoke_name(path):
        raise SystemExit(
            "Smoke tokenizers are forbidden in native MAGI chat/generate runtime. "
            f"tokenizer_id={tokenizer.tokenizer_id!r} path={path}. "
            "Pass a production tokenizer artifact, or use scripts/dev/smoke_chat.py."
        )


def _read_train_meta(checkpoint: Path) -> dict[str, Any] | None:
    if checkpoint.is_dir():
        meta_path = checkpoint / "train_meta.json"
    elif checkpoint.name == "model.safetensors":
        meta_path = checkpoint.parent / "train_meta.json"
    else:
        meta_path = checkpoint.parent / "train_meta.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def resolve_checkpoint_weights(checkpoint: Path, *, root: Path) -> Path:
    ckpt = checkpoint if checkpoint.is_absolute() else (root / checkpoint).resolve()
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    weights = ckpt / "model.safetensors" if ckpt.is_dir() else ckpt
    if not weights.exists():
        raise SystemExit(f"missing model.safetensors: {weights}")
    return weights


def resolve_tokenizer_path(
    *,
    explicit: Path | None,
    checkpoint: Path,
    config_path: Path,
    root: Path,
) -> Path:
    """Resolve tokenizer artifact: CLI > checkpoint meta > model YAML artifact."""
    if explicit is not None:
        path = explicit if explicit.is_absolute() else (root / explicit).resolve()
        if not path.exists():
            raise SystemExit(f"tokenizer not found: {path}")
        return path

    meta = _read_train_meta(checkpoint if checkpoint.is_absolute() else (root / checkpoint).resolve())
    if meta is not None:
        for key in ("tokenizer_path", "tokenizer_artifact"):
            raw = meta.get(key)
            if raw is None and isinstance(meta.get("manifest"), dict):
                raw = meta["manifest"].get(key)
            if raw is None and isinstance(meta.get("checkpoint"), dict):
                raw = meta["checkpoint"].get(key)
            if not raw:
                continue
            path = Path(str(raw))
            if not path.is_absolute():
                path = (root / path).resolve()
            if path.exists():
                return path

    raw_cfg = load_simple_yaml(config_path)
    artifact = (raw_cfg.get("tokenizer") or {}).get("artifact")
    if artifact:
        path = Path(str(artifact))
        if not path.is_absolute():
            path = (root / path).resolve()
        if path.exists():
            return path

    raise SystemExit(
        "MAGI tokenizer is required. "
        "Provide --tokenizer <artifact.json>, or store tokenizer_path in train_meta.json, "
        "or set tokenizer.artifact in the model config. "
        "Smoke tokenizers are forbidden in native chat runtime."
    )


def load_production_tokenizer(
    *,
    explicit: Path | None,
    checkpoint: Path,
    config_path: Path,
    root: Path,
    expected_vocab_size: int,
) -> tuple[MagiByteBPETokenizer, Path]:
    path = resolve_tokenizer_path(
        explicit=explicit,
        checkpoint=checkpoint,
        config_path=config_path,
        root=root,
    )
    tokenizer = load_tokenizer(path)
    assert_production_tokenizer(tokenizer, path=path)
    if tokenizer.vocab_size != expected_vocab_size:
        raise SystemExit(
            f"tokenizer vocab {tokenizer.vocab_size} != model vocab {expected_vocab_size} ({path})"
        )
    return tokenizer, path


def context_limit(cfg) -> int:
    return int(cfg.infer_context or cfg.train_context or 2048)


def build_chat_input_ids(
    tokenizer: MagiByteBPETokenizer,
    *,
    identity: str,
    turns: list[tuple[str, str]],
    max_prompt_tokens: int,
) -> list[int]:
    """Token-aware chat packing: keep identity + newest turns within budget.

    turns: ordered oldest→newest as (role, text) where role in {user, magi}.
    """
    if max_prompt_tokens < 8:
        raise ValueError("max_prompt_tokens too small")

    suffix = "\nMAGI:"
    suffix_ids = tokenizer.encode(suffix, add_bos=False, add_eos=False)
    identity_ids = tokenizer.encode(identity + "\n", add_bos=True, add_eos=False)

    reserved = len(identity_ids) + len(suffix_ids)
    if reserved >= max_prompt_tokens:
        raise SystemExit(
            f"identity+suffix ({reserved} tokens) exceeds prompt budget {max_prompt_tokens}"
        )
    budget = max_prompt_tokens - reserved

    turn_blocks: list[list[int]] = []
    for role, text in turns:
        if role == "user":
            block = f"User: {text}\n"
        elif role == "magi":
            block = f"MAGI: {text}\n"
        else:
            raise ValueError(f"unknown role {role!r}")
        turn_blocks.append(tokenizer.encode(block, add_bos=False, add_eos=False))

    selected: list[list[int]] = []
    used = 0
    for block in reversed(turn_blocks):
        if used + len(block) > budget:
            break
        selected.append(block)
        used += len(block)
    selected.reverse()

    ids = list(identity_ids)
    for block in selected:
        ids.extend(block)
    ids.extend(suffix_ids)
    return ids


def load_production_model_bundle(
    *,
    config_path: Path,
    checkpoint: Path,
    tokenizer_arg: Path | None,
    device,
    root: Path,
):
    """Config + required checkpoint + production tokenizer → MAGITransformer."""
    from magi.model import MAGITransformer
    from magi.train.checkpoint import load_model_safetensors

    cfg_path = config_path if config_path.is_absolute() else (root / config_path).resolve()
    cfg = load_model_config(cfg_path)
    assert_production_config(cfg_path, model_name=cfg.name)

    weights = resolve_checkpoint_weights(checkpoint, root=root)
    tokenizer, tok_path = load_production_tokenizer(
        explicit=tokenizer_arg,
        checkpoint=checkpoint if checkpoint.is_absolute() else (root / checkpoint).resolve(),
        config_path=cfg_path,
        root=root,
        expected_vocab_size=cfg.vocab_size,
    )

    model = MAGITransformer.from_config(cfg).to(device=device)
    load_model_safetensors(model, weights)
    model.eval()
    return cfg, model, tokenizer, weights, tok_path
