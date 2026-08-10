# CONFIG FORMAT

MAGI supports native YAML and HuggingFace `config.json`.

## Native YAML

Native configs remain source-of-truth for architecture planning:

- `configs/magi_35b_moe_v0.1.yaml`
- `configs/magi_400b_v0.1.yaml`
- `configs/magi_casual_v0.1.yaml`

Loaded by:

```python
from magi.config import load_model_config

cfg = load_model_config("configs/magi_35b_moe_v0.1.yaml")
```

## HF Config

```python
from magi.hf import MagiConfig

config = MagiConfig.from_native_yaml("configs/magi_35b_moe_v0.1.yaml")
config.save_pretrained("./hf-config")
```

Required identity fields:

```json
{
  "model_type": "magi",
  "architectures": ["MagiForCausalLM"],
  "architecture_version": "magi-architecture-v0.1",
  "checkpoint_version": "magi-checkpoint-v0.1",
  "config_version": "magi-config-v0.1",
  "serialization_version": "magi-serialization-v0.1"
}
```

## Roundtrip

```text
Native YAML → MagiConfig → config.json → MagiConfig → Native ModelConfig / YAML
```

Preserved architecture fields:

- `model_class`
- `d_model`, `n_layers`, `n_heads`, `n_kv_heads`, `d_head`
- dense / MoE FFN dimensions
- expert counts and routing width
- `vocab_size`
- `tied_embeddings`
- `rmsnorm_eps`, `rope_theta`, `bias`
- `train_context`, `infer_context`
- explicit version fields

Native MoE `top_k` maps to HF field `moe_top_k` so it does not collide with
Transformers sampling `GenerationConfig.top_k`. Conversion restores native `top_k`.

No silent field loss is allowed.

## Conversion API

```python
from magi.hf import native_config_to_hf, hf_config_to_native

hf = native_config_to_hf(native_cfg_or_yaml_path)
native = hf_config_to_native(hf)
```
