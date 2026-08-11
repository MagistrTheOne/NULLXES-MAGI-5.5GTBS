# CHECKPOINT FORMAT

MAGI checkpoint metadata is controlled by `magi.checkpoint.CheckpointManifest`.

## Versions

| Version | Stored In | Constant |
|---------|-----------|----------|
| Manifest Version | `CheckpointManifest.manifest_version` | `1.0` |
| Checkpoint Schema | `CheckpointManifest.checkpoint_schema_version` | `1` |
| Architecture Version | `MagiConfig.architecture_version` | `magi-architecture-v0.1` |
| Checkpoint Version | `MagiConfig.checkpoint_version` | `magi-checkpoint-v0.1` |
| Config Version | `MagiConfig.config_version` | `magi-config-v0.1` |
| Serialization Version | `MagiConfig.serialization_version` | `magi-serialization-v0.1` |

Versions are explicit. They are never inferred.

## Native Checkpoint Contract (`CheckpointManifest`)

Identity:

- `model_name`, `model_architecture`, `model_revision`
- `config_path`, `config_sha256`
- `total_parameters`, `active_parameters_per_token` (optional sanity)

Tokenizer:

- `tokenizer_id`, `tokenizer_sha256`

Data lineage:

- `dataset_manifest_id`, `dataset_manifest_sha256`, `mixture_id`

Training lineage:

- `train_config_sha256`, `run_id`
- `global_step`, `consumed_tokens`, `consumed_samples`

Topology / numeric:

- `world_size`, `parallelism`
- `parameter_dtype`, `compute_dtype`

Storage:

- `checkpoint_format`, `checkpoint_schema_version`, `manifest_version`

Inventory (writer fills **after** successful writes):

- `artifacts`: map of `{kind: {path, sha256, bytes, kind}}`
- `state_sections`: derived from keys actually present in `artifacts` (+ `config`, `tokenizer`)

`state_sections` is not a promise list. If `optimizer` is missing from `artifacts`, it was not written.

HF conversion never invents these fields. Use `build_hf_checkpoint_manifest(...)` with the same provenance inputs.

## Single-GPU Train Checkpoint (`magi_single_gpu_v0.3`)

Canonical weights: **`model.safetensors`**.

Layout:

```text
artifacts/.../
  model.safetensors
  train_meta.json
  optimizer.pt                 # when save_optimizer
  rng.pt                       # python + torch CPU + CUDA all-device
  step-000100/
    model.safetensors
    train_meta.json            # embeds CheckpointManifest
    optimizer.pt
    rng.pt
```

RNG bags: Python `random`, `torch.get_rng_state()`, `torch.cuda.get_rng_state_all()` when CUDA is present.

MoE: `expert_bias` lives in `model.state_dict()` / safetensors (persistent buffer). Acceptance: `tests/test_checkpoint_contract.py`.

Legacy `train.pt` (v0.1) and `magi_single_gpu_v0.2` weights still load via `load_train_checkpoint`.

## Resume gate

```text
TRAIN N → SAVE → new process → LOAD
  verify artifact sha256
  verify optimizer / rng present when claimed
  verify MoE expert_bias identical
→ TRAIN N+1
```

Gate: `python -m unittest tests.test_checkpoint_contract -v`
