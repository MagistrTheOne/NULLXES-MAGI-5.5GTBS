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
    router_z_loss: torch.Tensor
    gate_logits: torch.Tensor


class MoERouter(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_experts: int,
        top_k: int,
        bias: bool = False,
        *,
        aux_loss_free: bool = True,
    ) -> None:
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.aux_loss_free = aux_loss_free
        self.gate = nn.Linear(d_model, n_experts, bias=bias)
        # Aux-loss-free load-balance bias: not trained by Adam; updated from utilization.
        self.register_buffer("expert_bias", torch.zeros(n_experts), persistent=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, RouterStats]:
        logits = self.gate(x).float()
        route_logits = logits
        if self.aux_loss_free:
            route_logits = route_logits + self.expert_bias.to(dtype=logits.dtype)
        scores = torch.sigmoid(route_logits)
        scores = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1.0e-9)
        weights, indices = torch.topk(scores, k=self.top_k, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1.0e-9)
        entropy = -(scores * scores.clamp_min(1.0e-9).log()).sum(dim=-1).mean()
        tokens_per_expert = torch.bincount(indices.reshape(-1), minlength=self.n_experts).to(
            dtype=torch.float32
        )
        # Z-loss on raw gate logits (pre-bias) to keep router magnitudes stable.
        router_z_loss = logits.pow(2).mean()
        return (
            indices,
            weights.to(dtype=x.dtype),
            RouterStats(
                entropy=entropy,
                tokens_per_expert=tokens_per_expert,
                router_z_loss=router_z_loss,
                gate_logits=logits.detach(),
            ),
        )

    @torch.no_grad()
    def update_expert_bias(self, tokens_per_expert: torch.Tensor, update_rate: float) -> None:
        if not self.aux_loss_free or update_rate <= 0:
            return
        counts = tokens_per_expert.to(device=self.expert_bias.device, dtype=torch.float32)
        mean = counts.mean().clamp_min(1.0e-9)
        # Overused experts get negative bias; underused get positive.
        error = counts / mean - 1.0
        self.expert_bias.add_(-float(update_rate) * error.sign())


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
        self.router_z_loss_coeff = float(cfg.router_z_loss_coeff or 0.0)
        self.bias_update_rate = float(cfg.moe_bias_update_rate or 0.0)
        aux_free = (cfg.moe_load_balance or "aux_loss_free_bias") == "aux_loss_free_bias"
        self.router = MoERouter(
            cfg.d_model,
            cfg.n_routed_experts,
            cfg.top_k,
            bias=cfg.bias,
            aux_loss_free=aux_free,
        )
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

    def router_aux_loss(self) -> torch.Tensor | None:
        if self.last_router_stats is None or self.router_z_loss_coeff <= 0:
            return None
        return self.router_z_loss_coeff * self.last_router_stats.router_z_loss

    @torch.no_grad()
    def update_load_balance_bias(self) -> None:
        if self.last_router_stats is None:
            return
        self.router.update_expert_bias(self.last_router_stats.tokens_per_expert, self.bias_update_rate)
