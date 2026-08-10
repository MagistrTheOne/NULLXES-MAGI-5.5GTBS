"""Torch import boundary for MAGI model code."""

from __future__ import annotations


def require_torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "MAGI model runtime requires PyTorch. Architecture validators do not, "
            "but model init/forward must run in a PyTorch environment."
        ) from exc
    return torch
