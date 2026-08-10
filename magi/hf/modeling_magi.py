"""HuggingFace CausalLM wrapper for native MAGITransformer."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

try:
    from magi.hf.configuration_magi import MagiConfig
    from magi.hf.generation import build_generation_config
    from magi.hf.serialization import save_config_bundle
except ModuleNotFoundError:  # checkpoint remote-code layout (flat module names)
    from configuration_magi import MagiConfig  # type: ignore
    from generation import build_generation_config  # type: ignore

    def save_config_bundle(config: MagiConfig, output_dir):  # type: ignore
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        config.save_pretrained(output)
        build_generation_config(config).save_pretrained(output)


from magi.model import MAGITransformer


def _import_hf_model_bases():
    """Resolve PreTrainedModel / GenerationMixin across transformers 4.x and 5.x."""
    try:
        from transformers.modeling_outputs import CausalLMOutputWithPast
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "magi.hf.modeling_magi requires transformers.modeling_outputs.CausalLMOutputWithPast"
        ) from exc

    try:
        from transformers import PreTrainedModel
    except Exception:
        try:
            from transformers.modeling_utils import PreTrainedModel
        except Exception as utils_exc:
            raise ImportError(
                "magi.hf.modeling_magi could not import PreTrainedModel; "
                "transformers with a compatible torch backend is required"
            ) from utils_exc

    try:
        from transformers import GenerationMixin
    except Exception:
        try:
            from transformers.generation import GenerationMixin
        except Exception:
            try:
                from transformers.generation.utils import GenerationMixin
            except Exception as gen_exc:
                raise ImportError(
                    "magi.hf.modeling_magi could not import GenerationMixin; "
                    "transformers>=5 needs torch>=2.5 (or use transformers==4.57.1 on Colab)"
                ) from gen_exc

    # DummyObject backends (torch disabled inside transformers) are not usable.
    module_name = getattr(PreTrainedModel, "__module__", "")
    if module_name.endswith("dummy_pt_objects"):
        raise ImportError(
            "transformers disabled its PyTorch backend (often torch<2.5 with transformers>=5). "
            "Install torch>=2.5 or pin transformers==4.57.1"
        )

    return PreTrainedModel, GenerationMixin, CausalLMOutputWithPast


PreTrainedModel, GenerationMixin, CausalLMOutputWithPast = _import_hf_model_bases()


class _TiedLMHead(nn.Module):
    """Linear view over tied embedding weights for HF embedding APIs."""

    def __init__(self, embedding: nn.Embedding) -> None:
        super().__init__()
        self.embedding = embedding

    @property
    def weight(self) -> torch.nn.Parameter:
        return self.embedding.weight

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return F.linear(hidden_states, self.embedding.weight)


def _past_length(past_key_values: Any) -> int:
    if past_key_values is None:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    if len(past_key_values) == 0:
        return 0
    first = past_key_values[0]
    if first is None or first[0] is None:
        return 0
    return int(first[0].shape[-2])


def _is_cache_object(past_key_values: Any) -> bool:
    return past_key_values is not None and (
        hasattr(past_key_values, "to_legacy_cache")
        or hasattr(past_key_values, "update")
        or hasattr(past_key_values, "layers")
        or hasattr(past_key_values, "key_cache")
    )


def _to_legacy_past(past_key_values: Any) -> Any:
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "get_seq_length") and int(past_key_values.get_seq_length()) == 0:
        return None
    if hasattr(past_key_values, "to_legacy_cache"):
        legacy = past_key_values.to_legacy_cache()
        if not legacy or legacy[0] is None or legacy[0][0] is None:
            return None
        return legacy
    # transformers>=5 DynamicCache: layers[i].keys / .values
    layers = getattr(past_key_values, "layers", None)
    if layers is not None:
        legacy = []
        for layer in layers:
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if keys is None or values is None:
                return None
            legacy.append((keys, values))
        return tuple(legacy) if legacy else None
    # Older DynamicCache: key_cache / value_cache lists
    key_cache = getattr(past_key_values, "key_cache", None)
    value_cache = getattr(past_key_values, "value_cache", None)
    if key_cache is not None and value_cache is not None:
        if not key_cache or key_cache[0] is None:
            return None
        return tuple((k, v) for k, v in zip(key_cache, value_cache))
    if not past_key_values or past_key_values[0] is None or past_key_values[0][0] is None:
        return None
    return past_key_values


def _to_dynamic_past(past_key_values: Any, config: Any | None = None) -> Any:
    if past_key_values is None:
        return None
    from transformers import DynamicCache

    if hasattr(DynamicCache, "from_legacy_cache"):
        return DynamicCache.from_legacy_cache(past_key_values)

    # transformers>=5 removed from_legacy_cache; rebuild via update().
    try:
        cache = DynamicCache(config=config) if config is not None else DynamicCache()
    except TypeError:
        cache = DynamicCache()
    for layer_idx, layer_past in enumerate(past_key_values):
        key_states, value_states = layer_past
        cache.update(key_states, value_states, layer_idx)
    return cache


class MagiForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MagiConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _supports_cache_class = True
    _no_split_modules = ["TransformerBlock", "MoELayer"]
    _keys_to_ignore_on_load_unexpected = [r"lm_head\.weight"]

    def __init__(self, config: MagiConfig) -> None:
        super().__init__(config)
        self.model = MAGITransformer(config.to_native_config())
        self.generation_config = build_generation_config(config)
        self.post_init()
        self.generation_config = build_generation_config(self.config)

    def _init_weights(self, module: nn.Module) -> None:
        return

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        past_key_values: Any | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast | tuple[Any, ...]:
        if input_ids is None:
            raise ValueError("MagiForCausalLM.forward requires input_ids")
        # transformers>=5 may pass cache_position / logits_to_keep / etc. Native path uses
        # attention_mask + position_ids; unused generation kwargs are intentionally ignored.
        del cache_position, kwargs
        if return_dict is None:
            return_dict = bool(getattr(self.config, "return_dict", True))
        output_attentions = self.config.output_attentions if output_attentions is None else output_attentions
        output_hidden_states = (
            self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        )
        if use_cache is None:
            use_cache = False if self.training else bool(getattr(self.config, "use_cache", True))
        want_dynamic = _is_cache_object(past_key_values) or self._supports_cache_class
        legacy_past = _to_legacy_past(past_key_values)

        native = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=legacy_past,
            use_cache=bool(use_cache),
            output_attentions=bool(output_attentions),
            output_hidden_states=bool(output_hidden_states),
            return_dict=True,
        )
        logits = native.logits
        past_out = native.past_key_values
        if use_cache and past_out is not None and want_dynamic:
            past_out = _to_dynamic_past(past_out, config=self.config)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            if attention_mask is not None and attention_mask.shape == labels.shape:
                shift_labels = shift_labels.masked_fill(attention_mask[:, 1:] == 0, -100)
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )
        if not return_dict:
            values = (logits, past_out, native.hidden_states, native.attentions)
            return ((loss,) + values) if loss is not None else values
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_out,
            hidden_states=native.hidden_states,
            attentions=native.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Any | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(dim=-1) - 1
            position_ids = position_ids.masked_fill(attention_mask == 0, 0)
        if _past_length(past_key_values) > 0:
            input_ids = input_ids[:, -1:]
            if position_ids is not None:
                position_ids = position_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "use_cache": True if use_cache is None else use_cache,
        }

    def _reorder_cache(self, past_key_values: Any, beam_idx: torch.Tensor) -> Any:
        if past_key_values is None:
            return None
        if hasattr(past_key_values, "reorder_cache"):
            past_key_values.reorder_cache(beam_idx)
            return past_key_values
        legacy = _to_legacy_past(past_key_values)
        reordered: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_past in legacy:
            reordered.append(
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                )
            )
        return _to_dynamic_past(tuple(reordered), config=self.config)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.token_embedding

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.token_embedding = value
        self.config.vocab_size = int(value.num_embeddings)
        self.model.cfg = replace(self.model.cfg, vocab_size=int(value.num_embeddings))
        if self.config.tied_embeddings:
            self.tie_weights()

    def get_output_embeddings(self) -> nn.Module:
        if self.model.lm_head is not None:
            return self.model.lm_head
        return _TiedLMHead(self.model.token_embedding)

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        if self.config.tied_embeddings:
            if isinstance(new_embeddings, _TiedLMHead):
                self.set_input_embeddings(new_embeddings.embedding)
                return
            if isinstance(new_embeddings, nn.Embedding):
                self.set_input_embeddings(new_embeddings)
                return
            if isinstance(new_embeddings, nn.Linear):
                emb = nn.Embedding(
                    new_embeddings.out_features,
                    new_embeddings.in_features,
                    device=new_embeddings.weight.device,
                    dtype=new_embeddings.weight.dtype,
                )
                with torch.no_grad():
                    emb.weight.copy_(new_embeddings.weight)
                self.set_input_embeddings(emb)
                return
            raise TypeError("Tied MAGI output embeddings must expose embedding weights")
        if not isinstance(new_embeddings, nn.Linear):
            raise TypeError("Untied MAGI output embeddings must be an nn.Linear")
        self.model.lm_head = new_embeddings

    def tie_weights(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
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
            if new_num_tokens > rows:
                if mean_resizing:
                    mean = old_embedding.weight.mean(dim=0)
                    new_embedding.weight[rows:].copy_(mean.unsqueeze(0).expand(new_num_tokens - rows, -1))
                else:
                    nn.init.normal_(new_embedding.weight[rows:], mean=0.0, std=0.02)
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
                if new_num_tokens > rows:
                    if mean_resizing:
                        mean = old_head.weight.mean(dim=0)
                        new_head.weight[rows:].copy_(mean.unsqueeze(0).expand(new_num_tokens - rows, -1))
                    else:
                        nn.init.normal_(new_head.weight[rows:], mean=0.0, std=0.02)
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
        raise RuntimeError(
            "MagiForCausalLM does not support gradient checkpointing yet. "
            "supports_gradient_checkpointing=False by contract."
        )

    def gradient_checkpointing_disable(self) -> None:
        return

    def save_pretrained(self, save_directory: str | Path | None = None, *args: Any, **kwargs: Any):
        if save_directory is None:
            raise ValueError("save_directory is required")
        output = Path(save_directory)
        output.mkdir(parents=True, exist_ok=True)
        result = super().save_pretrained(output, *args, **kwargs)
        save_config_bundle(self.config, output)
        build_generation_config(self.config).save_pretrained(output)
        _export_hf_code_files(output)
        return result


def _export_hf_code_files(output_dir: Path) -> None:
    """Export short-name HF modules so Auto*.from_pretrained(trust_remote_code=True) works."""
    import shutil

    import magi.hf.configuration_magi as configuration_magi
    import magi.hf.generation as generation_mod
    import magi.hf.modeling_magi as modeling_magi
    import magi.hf.versions as versions_mod

    for module in (configuration_magi, modeling_magi, generation_mod, versions_mod):
        source = Path(module.__file__)
        shutil.copy2(source, output_dir / source.name)


def _register_auto_model() -> bool:
    try:
        from transformers import AutoModelForCausalLM

        AutoModelForCausalLM.register(MagiConfig, MagiForCausalLM)
        return True
    except ValueError:
        mapping = getattr(AutoModelForCausalLM, "_model_mapping", None)
        if mapping is not None and mapping.get(MagiConfig) is not MagiForCausalLM:
            raise
        return True
    except Exception:
        # Dummy AutoModel / incomplete torch backend must not kill MagiForCausalLM import.
        return False


_AUTO_MODEL_REGISTERED = _register_auto_model()
