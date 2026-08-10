"""MAGI native decoder transformer."""

from __future__ import annotations

from magi.config import ModelConfig
from magi.model.layers import GQAAttention, RMSNorm, SwiGLU, init_magi_weights
from magi.model.moe import MoELayer
from magi.model.outputs import MagiModelOutput
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
        attn_out, present, weights = self.attn(
            self.attn_norm(x),
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, present, weights


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
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        use_cache: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = False,
    ) -> torch.Tensor | MagiModelOutput | tuple:
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must be integer token ids")
        batch, seq = input_ids.shape
        past_len = 0
        if past_key_values is not None:
            if len(past_key_values) == 0 or past_key_values[0] is None or past_key_values[0][0] is None:
                past_key_values = None
            else:
                if len(past_key_values) != len(self.blocks):
                    raise ValueError("past_key_values layer count does not match model depth")
                past_len = int(past_key_values[0][0].shape[-2])

        if attention_mask is not None and attention_mask.shape[-1] != past_len + seq:
            raise ValueError(
                f"attention_mask length {attention_mask.shape[-1]} != past+query {past_len + seq}"
            )
        if position_ids is None and attention_mask is not None and past_len == 0:
            # HF-compatible positions for padded batches: cumsum(mask) - 1 on valid tokens.
            position_ids = attention_mask.long().cumsum(dim=-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 0)
        elif position_ids is None:
            position_ids = torch.arange(
                past_len,
                past_len + seq,
                device=input_ids.device,
                dtype=torch.long,
            )[None, :].expand(batch, -1)
        elif position_ids.shape != (batch, seq):
            raise ValueError(f"position_ids shape {tuple(position_ids.shape)} != {(batch, seq)}")

        x = self.token_embedding(input_ids)
        hidden_states: list[torch.Tensor] = [x] if output_hidden_states else []
        presents: list[tuple[torch.Tensor, torch.Tensor]] = []
        attentions: list[torch.Tensor] = []

        for layer_idx, block in enumerate(self.blocks):
            layer_past = None if past_key_values is None else past_key_values[layer_idx]
            x, present, weights = block(
                x,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=layer_past,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            if use_cache:
                if present is None:
                    raise RuntimeError("use_cache=True but attention returned no present KV")
                presents.append(present)
            if output_attentions:
                if weights is None:
                    raise RuntimeError("output_attentions=True but attention weights missing")
                attentions.append(weights)
            if output_hidden_states:
                hidden_states.append(x)

        x = self.final_norm(x)
        if output_hidden_states:
            hidden_states.append(x)
        if self.lm_head is None:
            logits = x @ self.token_embedding.weight.t()
        else:
            logits = self.lm_head(x)

        output = MagiModelOutput(
            logits=logits,
            past_key_values=tuple(presents) if use_cache else None,
            hidden_states=tuple(hidden_states) if output_hidden_states else None,
            attentions=tuple(attentions) if output_attentions else None,
        )
        if return_dict:
            return output
        if not use_cache and not output_attentions and not output_hidden_states:
            return logits
        return (logits, output.past_key_values, output.hidden_states, output.attentions)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
