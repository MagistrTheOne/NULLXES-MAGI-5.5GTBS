# TOKENIZER ARCHITECTURE SPEC v0.1

**Program:** MAGI-5.5GTBS  
**Config:** [`configs/tokenizer_experiments_v0.1.yaml`](../configs/tokenizer_experiments_v0.1.yaml)  
**Default:** shared MAGI tokenizer, vocab 131072  
**From-zero:** trained on NULLXES-approved corpus only.

---

## 1. Mission

Tokenizer is a NULLXES component. It must support MAGI-35B, MAGI-400B, MAGI-5.5GTBS, and CASUAL without inheriting another model vocabulary.

---

## 2. Candidate Matrix

| Candidate | Algorithm | Vocab | Byte fallback | Status |
|-----------|-----------|------:|---------------|--------|
| `bpe_131k` | BPE | 131072 | yes | DEFAULT candidate |
| `unigram_131k` | Unigram | 131072 | yes | challenger |
| `byte_hybrid_131k` | byte-level hybrid | 131072 | yes | challenger |

Final tokenizer is selected by holdout measurements, not taste.

---

## 3. Required Holdouts

RU formal, RU informal, RU slang, EN formal, EN informal, Python, TypeScript, math/LaTeX, scientific text, profanity in context, emoji/punctuation, RU/EN code-switching.

---

## 4. Metrics

| Metric | Purpose |
|--------|---------|
| tokens/character | compression |
| tokens/word | morphology and language pressure |
| compression ratio | storage + context efficiency |
| code fragmentation | code usability |
| RU morphology fragmentation | Russian naturalness |
| math notation behavior | symbol stability |
| rare-symbol behavior | byte fallback quality |
| profanity fragmentation | CASUAL register quality |
| code-switch fertility | mixed RU/EN stability |

---

## 5. Rejection Gates

| Gate | Value |
|------|------:|
| `<unk>` max | 0.1% |
| RU informal fertility regression | ≤8% |
| Code fragmentation regression | ≤10% |
| Math symbol breakage | forbidden |
| Byte fallback | required |

If shared 131072 fails, create a tokenizer v0.2 decision package. Do not silently import foreign vocab.

---

## 6. Artifacts

| Artifact | Path |
|----------|------|
| model | `tokenizer/artifacts/magi_tokenizer_v0.1.model` |
| json | `tokenizer/artifacts/magi_tokenizer_v0.1.json` |
| report | `tokenizer/reports/tokenizer_eval_v0.1.md` |
| config | `configs/tokenizer_experiments_v0.1.yaml` |

Artifacts are future outputs; this phase defines architecture only.

---

## 7. Integration

All model configs reference vocab size 131072. Tokenizer freeze must happen before real shard packing and before MAGI-35B training.

---

## 8. v0.1 FREEZE

| Decision | Value |
|----------|-------|
| Foreign vocab | forbidden |
| Default vocab size | 131072 |
| Default candidate | BPE 131k with byte fallback |
| Selection basis | empirical holdout metrics |
