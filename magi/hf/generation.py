"""Generation configuration helpers for MAGI."""

from __future__ import annotations

from magi.hf.configuration_magi import MagiConfig

try:
    from transformers import GenerationConfig
except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
    raise RuntimeError("magi.hf.generation requires the optional transformers package") from exc


def build_generation_config(config: MagiConfig) -> GenerationConfig:
    max_length = config.infer_context or config.train_context or 2048
    return GenerationConfig(
        max_length=int(max_length),
        use_cache=False,
        bos_token_id=getattr(config, "bos_token_id", None),
        eos_token_id=getattr(config, "eos_token_id", None),
        pad_token_id=getattr(config, "pad_token_id", None),
    )
