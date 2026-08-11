# Responsibility

Owns MAGI tokenizer architecture, experiment matrix, artifact naming, and tokenizer evaluation reports.

## Bring-up tokenizer (engineering only)

| Field | Value |
|-------|-------|
| Status | **BRINGUP_ONLY** |
| ID | `magi_bringup_8k_v0.1` |
| Config | `configs/tokenizer_bringup_8k_v0.1.yaml` |
| Artifact | `tokenizer/artifacts/magi_bringup_8k_v0.1.json` |
| Vocab | 8192 |
| `production_pretraining_allowed` | **false** |
| `production_checkpoint_allowed` | **false** |
| Allowed uses | runtime encode/decode, unit gates, `--allow-runtime-probe` |
| Forbidden | MAGI BASE pretraining, production checkpoints |

## Production target — MAGI_TOKENIZER_V1 (not frozen)

Requires `MAGI_BASE_PILOT_v0.1` then sweep:

- algorithms: Byte-level BPE, Unigram (byte_hybrid disabled until concrete MAGI algorithm exists)
- vocab: 65536 / 98304 / 131072 / 163840 → **8 candidates**
- selection: intrinsic metrics + representative MoE learning probe
- freeze fields: id, vocab, specials, normalization, pretokenization, merges/model, corpus manifest SHA, artifact SHA256

Matrix: `configs/tokenizer_experiments_v0.1.yaml`  
Phase order: `data/program/MAGI_NEXT_PHASE_v0.1.yaml`

Expected after freeze:

- `tokenizer/artifacts/magi_tokenizer_v1.json`
- `tokenizer/artifacts/magi_tokenizer_v1.model`
- `tokenizer/reports/tokenizer_eval_v1.md`

Until freeze: MAGI-7B-MoE BASE / 35B / Casual training remain blocked.
