# MAGI T4 Smoke Tokenizer v0.1

Paired with `configs/magi_t4_smoke_v0.1.yaml`.

| Field | Value |
|-------|-------|
| algorithm | byte_bpe |
| vocab_size | 8192 |
| normalization | NFKC |
| artifact | `tokenizer/artifacts/magi_t4_smoke_v0.1.json` |
| seed | `tokenizer/data/t4_smoke_seed.txt` |

Gates:

- exact vocab 8192
- encode/decode roundtrip on seed lines
- vocab aligned to model `architecture.vocab_size`
