"""HF-free autoregressive generation for native MAGITransformer.

Chain (no external LLM):
  input_ids → MAGITransformer.forward → logits → sample → next token → …
"""

from __future__ import annotations

from dataclasses import dataclass

from magi.model.transformer import MAGITransformer
from magi.model.torch_runtime import require_torch

torch = require_torch()


@dataclass(frozen=True)
class GenerateConfig:
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    do_sample: bool = False
    eos_token_id: int | None = None
    pad_token_id: int | None = None
    # Extra stop ids (e.g. future <|end_turn|>) — any match ends the turn.
    stop_token_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class GenerateResult:
    """Full sequences plus prompt length for newly-generated-token decode."""

    sequences: torch.Tensor
    prompt_length: int

    @property
    def new_token_ids(self) -> torch.Tensor:
        return self.sequences[:, self.prompt_length :]


def _apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    if top_k <= 0 or top_k >= logits.size(-1):
        return logits
    values, _ = torch.topk(logits, k=top_k, dim=-1)
    cutoff = values[..., -1, None]
    return logits.masked_fill(logits < cutoff, torch.finfo(logits.dtype).min)


def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    probs = torch.softmax(sorted_logits, dim=-1)
    cumulative = torch.cumsum(probs, dim=-1)
    mask = cumulative > top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return torch.zeros_like(logits).scatter(-1, sorted_idx, sorted_logits)


def sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    do_sample: bool,
) -> torch.Tensor:
    """Sample or argmax from last-position vocab logits [B, V] → [B]."""
    if logits.dim() != 2:
        raise ValueError(f"expected [B, V] logits, got {tuple(logits.shape)}")
    if not do_sample or temperature <= 0:
        return torch.argmax(logits, dim=-1)
    scaled = logits.float() / max(float(temperature), 1.0e-5)
    scaled = _apply_top_k(scaled, top_k)
    scaled = _apply_top_p(scaled, top_p)
    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.no_grad()
def generate(
    model: MAGITransformer,
    input_ids: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    config: GenerateConfig | None = None,
) -> GenerateResult:
    """Autoregressive decode using MAGI weights only (no Transformers / no APIs)."""
    cfg = config or GenerateConfig()
    if input_ids.dim() != 2:
        raise ValueError("input_ids must be [batch, seq]")
    if not isinstance(model, MAGITransformer):
        raise TypeError(
            f"generate() requires magi.model.MAGITransformer, got {type(model).__name__}"
        )
    model.eval()
    device = input_ids.device
    prompt_length = int(input_ids.shape[-1])
    generated = input_ids
    if attention_mask is None:
        if cfg.pad_token_id is None:
            attention_mask = torch.ones_like(generated)
        else:
            attention_mask = (generated != cfg.pad_token_id).long()

    past = None
    finished = torch.zeros(generated.size(0), dtype=torch.bool, device=device)

    for _ in range(int(cfg.max_new_tokens)):
        if past is None:
            step_ids = generated
            step_mask = attention_mask
            step_pos = None
        else:
            step_ids = generated[:, -1:]
            step_mask = attention_mask
            past_len = past[0][0].shape[-2]
            step_pos = torch.full(
                (generated.size(0), 1),
                past_len,
                dtype=torch.long,
                device=device,
            )

        out = model(
            step_ids,
            attention_mask=step_mask,
            position_ids=step_pos,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = out.past_key_values
        next_logits = out.logits[:, -1, :]
        next_ids = sample_next_token(
            next_logits,
            temperature=cfg.temperature,
            top_k=cfg.top_k,
            top_p=cfg.top_p,
            do_sample=cfg.do_sample,
        )
        stop_ids = set(int(x) for x in cfg.stop_token_ids)
        if cfg.eos_token_id is not None:
            stop_ids.add(int(cfg.eos_token_id))
        if stop_ids:
            fill = int(cfg.eos_token_id) if cfg.eos_token_id is not None else next(iter(stop_ids))
            for batch_idx in range(next_ids.size(0)):
                if bool(finished[batch_idx].item()):
                    next_ids[batch_idx] = fill
                    continue
                if int(next_ids[batch_idx].item()) in stop_ids:
                    finished[batch_idx] = True

        generated = torch.cat([generated, next_ids.unsqueeze(-1)], dim=-1)
        ones = torch.ones((generated.size(0), 1), dtype=attention_mask.dtype, device=device)
        attention_mask = torch.cat([attention_mask, ones], dim=-1)
        if bool(finished.all().item()):
            break
    return GenerateResult(sequences=generated, prompt_length=prompt_length)
