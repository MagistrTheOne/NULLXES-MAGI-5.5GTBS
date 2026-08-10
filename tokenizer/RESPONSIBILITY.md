# Responsibility

Owns MAGI tokenizer architecture, experiment matrix, artifact naming, and tokenizer evaluation reports.

## Production path

Defines the interface and outputs for:

- BPE / Unigram / byte-hybrid candidates;
- vocab size 131072;
- RU/EN/code/math/profanity/code-switch holdouts;
- fertility and fragmentation gates;
- tokenizer artifact paths consumed by shard builders.

Authoritative spec: `docs/TOKENIZER_ARCHITECTURE_SPEC_v0.1.md`.
Experiment matrix: `configs/tokenizer_experiments_v0.1.yaml`.

## Smoke path (T4)

Hardware smoke tokenizer for Colab/Tesla T4:

- config: `configs/tokenizer_t4_smoke_v0.1.yaml`
- paired model: `configs/magi_t4_smoke_v0.1.yaml`
- implementation: `magi/tokenizer/byte_bpe.py`
- artifact: `tokenizer/artifacts/magi_t4_smoke_v0.1.json`
- runner: `scripts/t4_smoke_run.py`

Smoke tokenizer is not a substitute for the 131k production candidate matrix.
