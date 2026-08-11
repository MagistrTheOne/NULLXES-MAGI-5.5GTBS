# CHECKPOINT FORMAT

MAGI checkpoint metadata is controlled by `magi.checkpoint.CheckpointManifest`.

## Versions

| Version | Stored In | Constant |
|---------|-----------|----------|
| Architecture Version | `MagiConfig.architecture_version` | `magi-architecture-v0.1` |
| Checkpoint Version | `MagiConfig.checkpoint_version` | `magi-checkpoint-v0.1` |
| Config Version | `MagiConfig.config_version` | `magi-config-v0.1` |
| Serialization Version | `MagiConfig.serialization_version` | `magi-serialization-v0.1` |

Versions are explicit. They are never inferred.

## Native Checkpoint Contract

`CheckpointManifest` fields:

- `model_name`
- `config_path`
- `config_sha256`
- `tokenizer_id`
- `tokenizer_sha256`
- `checkpoint_format`
- `parallelism`
- `state_sections`

HF conversion never invents these fields. Use `build_hf_checkpoint_manifest(...)` with the same provenance inputs.

## HF Save

```python
model.save_pretrained(path)
```

Produces:

- `config.json`
- `generation_config.json`
- `model.safetensors` when safetensors is available
- otherwise `pytorch_model.bin`

## State Dict Mapping

Native keys become HF keys by prefixing `model.`:

```text
token_embedding.weight → model.token_embedding.weight
blocks.0.attn.q_proj.weight → model.blocks.0.attn.q_proj.weight
```

HF → native removes the prefix.

Helpers:

- `native_state_dict_to_hf`
- `hf_state_dict_to_native`
- `magi.hf.serialization.save_state_dict`
- `magi.hf.serialization.load_state_dict`

## Single-GPU Train Checkpoint (`magi_single_gpu_v0.2`)

Canonical weights: **`model.safetensors`**.

Layout:

```text
artifacts/.../
  model.safetensors          # latest
  train_meta.json
  step-000100/
    model.safetensors
    train_meta.json
    optimizer.pt             # optional (--save-optimizer)
```

Legacy `train.pt` (v0.1) still loads via `load_train_checkpoint`.
