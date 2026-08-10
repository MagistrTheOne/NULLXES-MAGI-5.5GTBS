"""AutoClass registration for MAGI HuggingFace interoperability."""

from __future__ import annotations


def register_magi_auto_classes() -> bool:
    """Register MAGI with Transformers Auto classes when available."""
    try:
        from transformers import AutoConfig, AutoModelForCausalLM

        from magi.hf.configuration_magi import MagiConfig
        from magi.hf.modeling_magi import MagiForCausalLM
    except Exception:
        return False

    try:
        AutoConfig.register(MagiConfig.model_type, MagiConfig)
    except ValueError:
        registered = AutoConfig.for_model(MagiConfig.model_type)
        if registered is not MagiConfig:
            raise
    try:
        AutoModelForCausalLM.register(MagiConfig, MagiForCausalLM)
    except ValueError:
        mapping = getattr(AutoModelForCausalLM, "_model_mapping", None)
        if mapping is None or mapping.get(MagiConfig) is not MagiForCausalLM:
            raise
    return True


HF_AUTO_REGISTERED = register_magi_auto_classes()
