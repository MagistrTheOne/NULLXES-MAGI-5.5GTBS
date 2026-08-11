"""Core transformer layers for MAGI."""

from __future__ import annotations

import math

from magi.config import ModelConfig
from magi.model.torch_runtime import require_torch

torch = require_torch()
nn = torch.nn
F = torch.nn.functional

try:
    from flash_attn import flash_attn_func as _flash_attn_func
except Exception:  # pragma: no cover - optional CUDA extension
    _flash_attn_func = None


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

    def forward(self, position_ids: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        inv_freq = self.inv_freq.to(device=device)
        flat = position_ids.to(device=device, dtype=inv_freq.dtype).unsqueeze(-1)
        freqs = flat * inv_freq
        cos = freqs.cos()[:, :, None, :]
        sin = freqs.sin()[:, :, None, :]
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


def build_attention_bias(
    *,
    attention_mask: torch.Tensor | None,
    batch: int,
    query_len: int,
    past_len: int,
    kv_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Additive SDPA bias [B, 1, Q, KV], or None for pure causal full-sequence.

    Never calls .item() / GPU sync.
    """
    needs_explicit_causal = past_len > 0 or query_len != kv_len
    if attention_mask is None and not needs_explicit_causal:
        return None

    if attention_mask is not None:
        if attention_mask.dim() != 2:
            raise ValueError("attention_mask must have shape [batch, kv_len]")
        if attention_mask.shape != (batch, kv_len):
            raise ValueError(
                f"attention_mask shape {tuple(attention_mask.shape)} != {(batch, kv_len)}"
            )

    blocked = torch.zeros((batch, 1, query_len, kv_len), dtype=torch.bool, device=device)
    q_pos = torch.arange(past_len, past_len + query_len, device=device)[:, None]
    k_pos = torch.arange(kv_len, device=device)[None, :]
    blocked = blocked | (k_pos > q_pos)

    if attention_mask is not None:
        pad_blocked = attention_mask.to(device=device) == 0
        blocked = blocked | pad_blocked[:, None, None, :]

    bias = torch.zeros((batch, 1, query_len, kv_len), dtype=dtype, device=device)
    bias = bias.masked_fill(blocked, torch.finfo(dtype).min)
    return bias


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

    def forward(
        self,
        x: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None, torch.Tensor | None]:
        batch, seq, _ = x.shape
        past_len = 0 if past_key_value is None else int(past_key_value[0].shape[-2])
        if position_ids is None:
            position_ids = torch.arange(
                past_len,
                past_len + seq,
                device=x.device,
                dtype=torch.long,
            )[None, :].expand(batch, -1)

        q = self.q_proj(x).view(batch, seq, self.n_heads, self.d_head)
        k = self.k_proj(x).view(batch, seq, self.n_kv_heads, self.d_head)
        v = self.v_proj(x).view(batch, seq, self.n_kv_heads, self.d_head)

        cos, sin = self.rope(position_ids, x.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        k_cache = k.transpose(1, 2)
        v_cache = v.transpose(1, 2)
        if past_key_value is not None:
            k_cache = torch.cat((past_key_value[0], k_cache), dim=-2)
            v_cache = torch.cat((past_key_value[1], v_cache), dim=-2)
        present = (k_cache, v_cache) if use_cache else None

        k_attn = repeat_kv(k_cache.transpose(1, 2), self.kv_repeat).transpose(1, 2)
        v_attn = repeat_kv(v_cache.transpose(1, 2), self.kv_repeat).transpose(1, 2)
        q_attn = q.transpose(1, 2)
        kv_len = k_attn.shape[-2]

        attn_bias = build_attention_bias(
            attention_mask=attention_mask,
            batch=batch,
            query_len=seq,
            past_len=past_len,
            kv_len=kv_len,
            device=x.device,
            dtype=q_attn.dtype,
        )

        attn_weights = None
        if output_attentions:
            scale = 1.0 / math.sqrt(self.d_head)
            scores = torch.matmul(q_attn.float(), k_attn.float().transpose(-2, -1)) * scale
            if attn_bias is not None:
                scores = scores + attn_bias.float()
            elif seq == kv_len:
                causal = torch.ones((seq, kv_len), dtype=torch.bool, device=x.device).triu(1)
                scores = scores.masked_fill(causal, torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=-1).to(dtype=q_attn.dtype)
            attn_weights = weights
            y = torch.matmul(weights, v_attn)
        elif attn_bias is None and past_len == 0 and _flash_attn_func is not None and x.is_cuda:
            # flash-attn: [B, S, H, D]; GQA via distinct q/kv head counts when supported.
            try:
                y = _flash_attn_func(
                    q.contiguous(),
                    k_cache.transpose(1, 2).contiguous(),
                    v_cache.transpose(1, 2).contiguous(),
                    causal=True,
                )
                y = y.transpose(1, 2)
            except Exception:
                y = F.scaled_dot_product_attention(q_attn, k_attn, v_attn, is_causal=True)
        elif attn_bias is None:
            y = F.scaled_dot_product_attention(q_attn, k_attn, v_attn, is_causal=True)
        else:
            y = F.scaled_dot_product_attention(q_attn, k_attn, v_attn, attn_mask=attn_bias, is_causal=False)

        y = y.transpose(1, 2).contiguous().view(batch, seq, self.d_model)
        return self.o_proj(y), present, attn_weights


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
