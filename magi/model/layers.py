"""Core transformer layers for MAGI."""

from __future__ import annotations

import math

from magi.config import ModelConfig
from magi.model.torch_runtime import require_torch

torch = require_torch()
nn = torch.nn
F = torch.nn.functional


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1.0e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x_norm = x * torch.rsqrt(variance + self.eps).to(dtype=x.dtype)
        return x_norm * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE head dimension must be even")
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(positions, self.inv_freq.to(device=device))
        cos = freqs.cos()[None, :, None, :]
        sin = freqs.sin()[None, :, None, :]
        return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = torch.stack((x_even * cos - x_odd * sin, x_even * sin + x_odd * cos), dim=-1)
    return rotated.flatten(-2)


def repeat_kv(x: torch.Tensor, repeat: int) -> torch.Tensor:
    if repeat == 1:
        return x
    batch, seq, heads, head_dim = x.shape
    x = x[:, :, :, None, :].expand(batch, seq, heads, repeat, head_dim)
    return x.reshape(batch, seq, heads * repeat, head_dim)


class GQAAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.d_head = cfg.d_head
        self.d_model = cfg.d_model
        self.kv_repeat = cfg.n_heads // cfg.n_kv_heads
        self.q_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.d_head, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.d_head, bias=cfg.bias)
        self.o_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=cfg.bias)
        self.rope = RotaryEmbedding(cfg.d_head, cfg.rope_theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        q = self.q_proj(x).view(batch, seq, self.n_heads, self.d_head)
        k = self.k_proj(x).view(batch, seq, self.n_kv_heads, self.d_head)
        v = self.v_proj(x).view(batch, seq, self.n_kv_heads, self.d_head)
        cos, sin = self.rope(seq, x.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        k = repeat_kv(k, self.kv_repeat)
        v = repeat_kv(v, self.kv_repeat)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(batch, seq, self.d_model)
        return self.o_proj(y)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, bias: bool = False) -> None:
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=bias)
        self.w3 = nn.Linear(d_model, d_ff, bias=bias)
        self.w2 = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


def init_magi_weights(module: nn.Module, n_layers: int, emb_std: float = 0.02) -> None:
    residual_std = 0.02 / math.sqrt(2.0 * n_layers)
    if isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=emb_std)
    elif isinstance(module, nn.Linear):
        std = residual_std if module.out_features == module.in_features else emb_std
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, RMSNorm):
        nn.init.ones_(module.weight)
