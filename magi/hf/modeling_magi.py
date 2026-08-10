"""HuggingFace CausalLM wrapper for native MAGITransformer."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from magi.hf.configuration_magi import MagiConfig
from magi.hf.generation import build_generation_config
from magi.hf.serialization import save_config_bundle
from magi.model import MAGITransformer

try:
    from transformers import GenerationMixin, PreTrainedModel
    from transformers.modeling_outputs import CausalLMOutputWithPast
except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
    raise RuntimeError("magi.hf.modeling_magi requires the optional transformers package") from exc


class MagiForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MagiConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_cache_class = False
    _no_split_modules = ["TransformerBlock", "MoELayer"]
    _keys_to_ignore_on_load_unexpected = [r"lm_head.weight"]
    _keys_to_ignore_on_load_missing = [r"lm_head.weight"]

    def __init__(self, config: MagiConfig) -> None:
        super().__init__(config)
        self.model = MAGITransformer(config.to_native_config())
        self.gradient_checkpointing = False
        self.generation_config = build_generation_config(config)
        self.post_init()
        self.generation_config = build_generation_config(self.config)

    def _init_weights(self, module: nn.Module) -> None:
        # Native MAGITransformer already applies MAGI initialization.
        return

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast | tuple[Any, ...]:
        if input_ids is None:
            raise ValueError("MagiForCausalLM.forward requires input_ids")
        del attention_mask, past_key_values, use_cache, kwargs
        return_dict = self.config.use_return_dict if return_dict is None else return_dict
        output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
        output_hidden_states = (
            self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        )
        if self.gradient_checkpointing and self.training:

            def _forward_blocks(token_ids: torch.Tensor) -> torch.Tensor:
                return self.model(token_ids, output_hidden_states=False)

            logits = torch.utils.checkpoint.checkpoint(
                _forward_blocks,
                input_ids,
                use_reentrant=False,
            )
            hidden_states = None
        else:
            native_output = self.model(input_ids, output_hidden_states=bool(output_hidden_states))
            if output_hidden_states:
                logits, hidden_states = native_output
            else:
                logits = native_output
                hidden_states = None
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        if not return_dict:
            values = (logits, None, hidden_states, None if not output_attentions else None)
            return ((loss,) + values) if loss is not None else values
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=hidden_states,
            attentions=None,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Any | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del past_key_values, kwargs
        return {
            "input_ids": input_ids,
            "past_key_values": None,
            "attention_mask": attention_mask,
            "use_cache": False,
        }

    def _reorder_cache(self, past_key_values: Any, beam_idx: torch.Tensor) -> Any:
        del past_key_values, beam_idx
        return None

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.token_embedding

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.token_embedding = value
        self.config.vocab_size = int(value.num_embeddings)
        self.model.cfg = replace(self.model.cfg, vocab_size=int(value.num_embeddings))
        if self.config.tied_embeddings:
            self.tie_weights()

    def get_output_embeddings(self) -> nn.Module:
        return self.model.token_embedding if self.model.lm_head is None else self.model.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        if self.config.tied_embeddings:
            if not isinstance(new_embeddings, nn.Embedding):
                raise TypeError("Tied MAGI output embeddings must be an nn.Embedding")
            self.set_input_embeddings(new_embeddings)
        else:
            if not isinstance(new_embeddings, nn.Linear):
                raise TypeError("Untied MAGI output embeddings must be an nn.Linear")
            self.model.lm_head = new_embeddings

    def tie_weights(self) -> None:
        if self.config.tied_embeddings:
            self.model.lm_head = None
            self.config.tie_word_embeddings = True
            self.config.tied_embeddings = True

    def resize_token_embeddings(
        self,
        new_num_tokens: int | None = None,
        pad_to_multiple_of: int | None = None,
        mean_resizing: bool = True,
    ) -> nn.Embedding:
        del mean_resizing
        if new_num_tokens is None:
            return self.get_input_embeddings()
        if pad_to_multiple_of is not None and new_num_tokens % pad_to_multiple_of != 0:
            new_num_tokens = ((new_num_tokens // pad_to_multiple_of) + 1) * pad_to_multiple_of
        old_embedding = self.model.token_embedding
        new_embedding = nn.Embedding(
            new_num_tokens,
            old_embedding.embedding_dim,
            device=old_embedding.weight.device,
            dtype=old_embedding.weight.dtype,
        )
        rows = min(old_embedding.num_embeddings, new_num_tokens)
        with torch.no_grad():
            new_embedding.weight[:rows].copy_(old_embedding.weight[:rows])
        self.model.token_embedding = new_embedding
        if self.model.lm_head is not None:
            old_head = self.model.lm_head
            new_head = nn.Linear(
                old_head.in_features,
                new_num_tokens,
                bias=old_head.bias is not None,
                device=old_head.weight.device,
                dtype=old_head.weight.dtype,
            )
            with torch.no_grad():
                new_head.weight[:rows].copy_(old_head.weight[:rows])
                if old_head.bias is not None and new_head.bias is not None:
                    new_head.bias[:rows].copy_(old_head.bias[:rows])
            self.model.lm_head = new_head
        self.config.vocab_size = int(new_num_tokens)
        self.model.cfg = replace(self.model.cfg, vocab_size=int(new_num_tokens))
        if self.config.tied_embeddings:
            self.tie_weights()
        return self.model.token_embedding

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict[str, Any] | None = None) -> None:
        del gradient_checkpointing_kwargs
        self.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing = False

    def save_pretrained(self, save_directory: str | Path | None = None, *args: Any, **kwargs: Any):
        if save_directory is None:
            raise ValueError("save_directory is required")
        output = Path(save_directory)
        output.mkdir(parents=True, exist_ok=True)
        result = super().save_pretrained(output, *args, **kwargs)
        save_config_bundle(self.config, output)
        build_generation_config(self.config).save_pretrained(output)
        return result
