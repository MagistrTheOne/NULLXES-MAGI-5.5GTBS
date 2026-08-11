"""Runtime helpers for MAGI model bring-up and native inference."""

from magi.runtime.generate import GenerateConfig, GenerateResult, generate, sample_next_token

__all__ = ["GenerateConfig", "GenerateResult", "generate", "sample_next_token"]
