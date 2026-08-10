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
HF_IMPORT_ERROR: BaseException | None = None
MagiConfig = None  # type: ignore[assignment]
MagiForCausalLM = None  # type: ignore[assignment]


def _unavailable(*_args, **_kwargs):
    detail = ""
    if HF_IMPORT_ERROR is not None:
        detail = f" ({type(HF_IMPORT_ERROR).__name__}: {HF_IMPORT_ERROR})"
    raise ImportError(
        "magi.hf requires a working transformers install for this API"
        + detail
    ) from HF_IMPORT_ERROR


def require_hf():
    """Return MagiForCausalLM or raise with the original import failure."""
    if MagiForCausalLM is None or not HF_AVAILABLE:
        _unavailable()
    return MagiForCausalLM


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
except (ImportError, ModuleNotFoundError) as exc:
    HF_AVAILABLE = False
    HF_AUTO_REGISTERED = False
    HF_IMPORT_ERROR = exc
    MagiConfig = None  # type: ignore[assignment]
    MagiForCausalLM = None  # type: ignore[assignment]
    native_config_to_hf = _unavailable
    hf_config_to_native = _unavailable
    native_state_dict_to_hf = _unavailable
    hf_state_dict_to_native = _unavailable
    save_hf_config_from_native = _unavailable
    save_native_yaml_from_hf = _unavailable


__all__ = [
    "ARCHITECTURE_VERSION",
    "CHECKPOINT_VERSION",
    "CONFIG_VERSION",
    "SERIALIZATION_VERSION",
    "HF_AVAILABLE",
    "HF_AUTO_REGISTERED",
    "HF_IMPORT_ERROR",
    "MagiConfig",
    "MagiForCausalLM",
    "hf_config_to_native",
    "hf_state_dict_to_native",
    "native_config_to_hf",
    "native_state_dict_to_hf",
    "register_magi_auto_classes",
    "require_hf",
    "save_hf_config_from_native",
    "save_native_yaml_from_hf",
]
