# HF COMPATIBILITY

MAGI supports two loading paths.

## Native

```python
from magi.config import load_model_config
from magi.model import MAGITransformer

cfg = load_model_config("configs/magi_35b_moe_v0.1.yaml")
model = MAGITransformer.from_config(cfg)
```

Native remains canonical. HuggingFace is interoperability only.

## HuggingFace Transformers

```python
import magi  # registers Auto classes when transformers is installed
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("./checkpoint")
model = AutoModelForCausalLM.from_pretrained("./checkpoint")
model.save_pretrained("./checkpoint-out")
```

Without prior `import magi`, saved checkpoints still load via `auto_map`:

```python
from transformers import AutoConfig, AutoModelForCausalLM

config = AutoConfig.from_pretrained("./checkpoint", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("./checkpoint", trust_remote_code=True)
```

`magi.hf.configuration_magi` / `modeling_magi` self-register on import.

## Runtime contracts

| API | Status |
|-----|--------|
| `attention_mask` | applied in native GQA (pad + causal) |
| `position_ids` | supported; auto from mask when omitted |
| `use_cache` / `past_key_values` | real KV cache |
| `prepare_inputs_for_generation` | last-token cut + cache |
| `_reorder_cache` | beam reorder |
| `output_attentions` | real per-layer weights |
| gradient checkpointing | explicitly unsupported |

## Names

| Field | Value |
|-------|-------|
| `model_type` | `magi` |
| config class | `MagiConfig` |
| model class | `MagiForCausalLM` |
| native architecture | `MAGITransformer` |
| base model prefix | `model` |

## Package Layout

```text
magi/hf/
  configuration_magi.py
  modeling_magi.py
  auto.py
  convert.py
  serialization.py
  generation.py
  versions.py
```

Installable package: `pip install -e .` (`pyproject.toml`).

## Optional Dependency

`transformers` and `safetensors` are optional.

- Native MAGI imports without them.
- HF bridge activates only when `transformers` imports cleanly.
- Weight save prefers `model.safetensors` when `safetensors` is installed; otherwise `pytorch_model.bin`.

## Guarantees

- Native parameter accounting is unchanged.
- HF wrapper does not fork transformer code.
- Config architecture fields roundtrip losslessly Native ↔ HF.
- Checkpoint manifests remain authoritative for provenance metadata.
