"""Mixture-of-Experts layers for MAGI."""

from __future__ import annotations

from dataclasses import dataclass

from magi.config import ModelConfig
from magi.model.layers import SwiGLU
from magi.model.torch_runtime import require_torch

torch = require_torch()
nn = torch.nn


@dataclass(frozen=True)
class RouterStats:
    entropy: torch.Tensor
    tokens_per_expert: torch.Tensor


class MoERouter(nn.Module):
    def __init__(self, d_model: int, n_experts: int, top_k: int, bias: bool = False) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.gate = nn.Linear(d_model, n_experts, bias=bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, RouterStats]:
        logits = self.gate(x).float()
        scores = torch.sigmoid(logits)
        scores = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1.0e-9)
        weights, indices = torch.topk(scores, k=self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-9)
        entropy = -(scores * scores.clamp_min(1.0e-9).log()).sum(dim=-1).mean()
        tokens_per_expert = torch.bincount(indices.reshape(-1), minlength=self.n_experts)
        return indices, weights.to(dtype=x.dtype), RouterStats(entropy=entropy, tokens_per_expert=tokens_per_expert)


class MoELayer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if not cfg.is_moe:
            raise ValueError("MoELayer requires an MoE ModelConfig")
        assert cfg.n_routed_experts is not None
        assert cfg.n_shared_experts is not None
        assert cfg.top_k is not None
        assert cfg.d_ff_expert is not None
        self.n_routed_experts = cfg.n_routed_experts
        self.n_shared_experts = cfg.n_shared_experts
        self.top_k = cfg.top_k
        self.router = MoERouter(cfg.d_model, cfg.n_routed_experts, cfg.top_k, bias=cfg.bias)
        self.routed_experts = nn.ModuleList(
            [SwiGLU(cfg.d_model, cfg.d_ff_expert, cfg.bias) for _ in range(cfg.n_routed_experts)]
        )
        self.shared_experts = nn.ModuleList(
            [SwiGLU(cfg.d_model, cfg.d_ff_expert, cfg.bias) for _ in range(cfg.n_shared_experts)]
        )
        self.last_router_stats: RouterStats | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        flat = x.reshape(-1, original_shape[-1])
        indices, weights, stats = self.router(flat)
        output = torch.zeros_like(flat)

        for slot in range(self.top_k):
            expert_ids = indices[:, slot]
            expert_weights = weights[:, slot].unsqueeze(-1)
            for expert_id, expert in enumerate(self.routed_experts):
                mask = expert_ids == expert_id
                if mask.any():
                    output[mask] += expert(flat[mask]) * expert_weights[mask]

        if self.shared_experts:
            shared = torch.zeros_like(flat)
            for expert in self.shared_experts:
                shared = shared + expert(flat)
            output = output + shared / float(len(self.shared_experts))

        self.last_router_stats = stats
        return output.reshape(original_shape)
