"""MAGI native decoder transformer."""

from __future__ import annotations

from magi.config import ModelConfig
from magi.model.layers import GQAAttention, RMSNorm, SwiGLU, init_magi_weights
from magi.model.moe import MoELayer
from magi.model.torch_runtime import require_torch

torch = require_torch()
nn = torch.nn


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_index: int) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rmsnorm_eps)
        self.attn = GQAAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.rmsnorm_eps)
        self.is_moe = cfg.is_moe and cfg.n_dense_layers is not None and layer_index >= cfg.n_dense_layers
        if self.is_moe:
            self.ffn = MoELayer(cfg)
        else:
            self.ffn = SwiGLU(cfg.d_model, cfg.dense_ffn_dim, cfg.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class MAGITransformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        cfg.validate()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg, i) for i in range(cfg.n_layers)])
        self.final_norm = RMSNorm(cfg.d_model, cfg.rmsnorm_eps)
        if cfg.tied_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(lambda module: init_magi_weights(module, cfg.n_layers))

    @classmethod
    def from_config(cls, cfg: ModelConfig, device: str | torch.device | None = None) -> "MAGITransformer":
        if device is None:
            return cls(cfg)
        with torch.device(device):
            return cls(cfg)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        output_hidden_states: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must be integer token ids")
        x = self.token_embedding(input_ids)
        hidden_states: list[torch.Tensor] = [x] if output_hidden_states else []
        for block in self.blocks:
            x = block(x)
            if output_hidden_states:
                hidden_states.append(x)
        x = self.final_norm(x)
        if output_hidden_states:
            hidden_states.append(x)
        if self.lm_head is None:
            logits = x @ self.token_embedding.weight.t()
        else:
            logits = self.lm_head(x)
        if output_hidden_states:
            return logits, tuple(hidden_states)
        return logits

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
