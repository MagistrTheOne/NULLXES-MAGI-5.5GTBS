"""Optional HuggingFace Transformers interoperability for MAGI."""

from __future__ import annotations

from magi.hf.versions import (
    ARCHITECTURE_VERSION,
    CHECKPOINT_VERSION,
    CONFIG_VERSION,
    SERIALIZATION_VERSION,
)

HF_AVAILABLE = False
HF_AUTO_REGISTERED = False
MagiConfig = None  # type: ignore[assignment]
MagiForCausalLM = None  # type: ignore[assignment]


def _unavailable(*_args, **_kwargs):
    raise ImportError("magi.hf requires the optional transformers package")


def register_magi_auto_classes() -> bool:
    from magi.hf.auto import register_magi_auto_classes as _register

    return _register()


native_config_to_hf = _unavailable
hf_config_to_native = _unavailable
native_state_dict_to_hf = _unavailable
hf_state_dict_to_native = _unavailable
save_hf_config_from_native = _unavailable
save_native_yaml_from_hf = _unavailable

try:
    from magi.hf.configuration_magi import MagiConfig
    from magi.hf.convert import (
        hf_config_to_native,
        hf_state_dict_to_native,
        native_config_to_hf,
        native_state_dict_to_hf,
        save_hf_config_from_native,
        save_native_yaml_from_hf,
    )
    from magi.hf.modeling_magi import MagiForCausalLM

    HF_AVAILABLE = True
    HF_AUTO_REGISTERED = register_magi_auto_classes()
except (ImportError, ModuleNotFoundError):
    HF_AVAILABLE = False
    HF_AUTO_REGISTERED = False


__all__ = [
    "ARCHITECTURE_VERSION",
    "CHECKPOINT_VERSION",
    "CONFIG_VERSION",
    "SERIALIZATION_VERSION",
    "HF_AVAILABLE",
    "HF_AUTO_REGISTERED",
    "MagiConfig",
    "MagiForCausalLM",
    "hf_config_to_native",
    "hf_state_dict_to_native",
    "native_config_to_hf",
    "native_state_dict_to_hf",
    "register_magi_auto_classes",
    "save_hf_config_from_native",
    "save_native_yaml_from_hf",
]
