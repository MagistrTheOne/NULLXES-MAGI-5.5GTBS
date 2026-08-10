"""Dry initialization for MAGI models."""

from __future__ import annotations

from pathlib import Path

from magi.config import load_model_config
from magi.model import MAGITransformer


def dry_init_model(config_path: str | Path, device: str = "meta") -> tuple[MAGITransformer, int]:
    cfg = load_model_config(config_path)
    model = MAGITransformer.from_config(cfg, device=device)
    return model, model.parameter_count()
