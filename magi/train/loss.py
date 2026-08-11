"""Causal LM loss helpers for native MAGITransformer."""

from __future__ import annotations

from dataclasses import dataclass

from magi.model.moe import MoELayer
from magi.model.torch_runtime import require_torch

torch = require_torch()
F = torch.nn.functional


@dataclass(frozen=True)
class RouterTelemetry:
    entropy: float
    expert_load: list[float]
    dead_experts: int
    n_experts: int
    imbalance_ratio: float


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
    telemetry = collect_router_telemetry(model)
    if telemetry is None:
        return None
    return torch.tensor(telemetry.entropy)


def collect_router_telemetry(model: torch.nn.Module) -> RouterTelemetry | None:
    """Aggregate MoE router stats across MoE layers (entropy / load / dead experts)."""
    entropies: list[torch.Tensor] = []
    load_acc: torch.Tensor | None = None
    n_experts: int | None = None

    for module in model.modules():
        if not isinstance(module, MoELayer) or module.last_router_stats is None:
            continue
        stats = module.last_router_stats
        entropies.append(stats.entropy.detach().float())
        counts = stats.tokens_per_expert.detach().float()
        if load_acc is None:
            load_acc = counts.clone()
            n_experts = int(counts.numel())
        else:
            load_acc = load_acc + counts

    if not entropies or load_acc is None or n_experts is None:
        return None

    total_tokens = float(load_acc.sum().item())
    if total_tokens <= 0:
        load = [0.0] * n_experts
        dead = n_experts
        imbalance = float("inf")
    else:
        load = (load_acc / total_tokens).tolist()
        dead = int((load_acc == 0).sum().item())
        mean = total_tokens / float(n_experts)
        imbalance = float((load_acc.max() / max(mean, 1.0e-9)).item())

    return RouterTelemetry(
        entropy=float(torch.stack(entropies).mean().item()),
        expert_load=load,
        dead_experts=dead,
        n_experts=n_experts,
        imbalance_ratio=imbalance,
    )
