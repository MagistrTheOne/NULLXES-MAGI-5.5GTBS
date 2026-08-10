"""Causal LM loss helpers for native MAGITransformer."""

from __future__ import annotations

from magi.model.moe import MoELayer
from magi.model.torch_runtime import require_torch

torch = require_torch()
F = torch.nn.functional


def causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )


def collect_router_entropy(model: torch.nn.Module) -> torch.Tensor | None:
    entropies: list[torch.Tensor] = []
    for module in model.modules():
        if isinstance(module, MoELayer) and module.last_router_stats is not None:
            entropies.append(module.last_router_stats.entropy.detach())
    if not entropies:
        return None
    return torch.stack(entropies).mean()
