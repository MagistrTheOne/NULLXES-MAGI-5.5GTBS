"""Conversion helpers between native MAGI and HuggingFace formats."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from magi.config import ModelConfig, load_model_config
from magi.hf.configuration_magi import MagiConfig


def native_config_to_hf(config: ModelConfig | str | Path) -> MagiConfig:
    if isinstance(config, ModelConfig):
        return MagiConfig.from_native_config(config)
    if Path(config).suffix in {".yaml", ".yml"}:
        return MagiConfig.from_native_yaml(config)
    return MagiConfig.from_native_config(load_model_config(config))


def hf_config_to_native(config: MagiConfig) -> ModelConfig:
    return config.to_native_config()


def native_state_dict_to_hf(state_dict: Mapping[str, object]) -> dict[str, object]:
    return {f"model.{key}": value for key, value in state_dict.items()}


def hf_state_dict_to_native(state_dict: Mapping[str, object]) -> dict[str, object]:
    native: dict[str, object] = {}
    for key, value in state_dict.items():
        native[key[6:] if key.startswith("model.") else key] = value
    return native


def save_hf_config_from_native(native_yaml: str | Path, output_dir: str | Path) -> MagiConfig:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = native_config_to_hf(native_yaml)
    config.save_pretrained(output)
    return config


def save_native_yaml_from_hf(config: MagiConfig, output_path: str | Path) -> None:
    config.save_native_yaml(output_path)
