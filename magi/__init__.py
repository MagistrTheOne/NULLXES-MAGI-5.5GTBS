"""NULLXES MAGI native architecture package."""

from magi.config import ModelConfig, load_model_config

try:
    from magi.hf import HF_AUTO_REGISTERED, register_magi_auto_classes

    if not HF_AUTO_REGISTERED:
        HF_AUTO_REGISTERED = register_magi_auto_classes()
except Exception:
    HF_AUTO_REGISTERED = False

__all__ = ["HF_AUTO_REGISTERED", "ModelConfig", "load_model_config"]
