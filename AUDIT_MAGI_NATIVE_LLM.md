# AUDIT_MAGI_NATIVE_LLM.md

**Program:** MAGI-5.5GTBS / NULLXES  
**Auditor role:** Principal LLM Architect — executable-code audit only  
**Constraint:** no local test execution · no dependency install on audit host  
**Date:** 2026-08-11

---

## 1. EXECUTIVE VERDICT

| Gate | Status |
|------|--------|
| **NATIVE MAGI LLM** | **PARTIAL** |
| **EXTERNAL LLM DEPENDENCY** | **NO** |
| **REAL FROM-ZERO TRAINING PATH** | **YES** (single-GPU bring-up) |
| **REAL SPARSE-MOE MODEL** | **YES** |
| **REAL TOKENIZER** | **PARTIAL** (smoke Byte-BPE only; 131k production absent) |
| **REAL CAUSAL LOSS** | **YES** |
| **REAL BACKPROP** | **YES** (code path) |
| **REAL CHECKPOINT** | **YES** (`model.safetensors` + resume) |
| **REAL NATIVE GENERATION** | **YES** (HF-free `magi.runtime.generate`) |
| **CASUAL CHAT READY** | **PARTIAL** (REPL exists; needs trained checkpoint) |

### Verdict in one line

MAGI **is not a wrapper**. Executable Sparse-MoE decoder, from-scratch init, causal loss, optimizer step, and safetensors checkpoints exist. It is **not yet a finished native product LLM**: production tokenizer missing, MoE load-balance / z-loss declared in YAML but not executed, no `magi chat` identity path, generation depended on Transformers `GenerationMixin`.

---

## 2. P0 BLOCKERS

**None found for “external intelligence owns MAGI output”.**

Searched repo for OpenAI / Anthropic / Grok / Gemini / Bedrock / Together / OpenRouter / Ollama / remote completion SDKs. Occurrences of `from_pretrained` are **HF load of MAGI checkpoints** (`MagiForCausalLM` wrapping `MAGITransformer`), not foreign backbones.

| Candidate P0 | Result |
|--------------|--------|
| External LLM generates MAGI text | **ABSENT** |
| Pretrained foreign model as backbone | **ABSENT** |
| No native forward / LM head | **ABSENT** (exists) |
| Generation simulated / hardcoded | **ABSENT** (logits → sample) |

If a future path reintroduces API cognition into runtime, reclassify as **CRITICAL ARCHITECTURAL FAILURE**.

---

## 3. P1 BLOCKERS

| ID | Blocker | Evidence |
|----|---------|----------|
| P1-01 | No HF-free autoregressive product path / no `magi chat` | **FIXED** — `magi/runtime/generate.py`, `scripts/magi_chat.py` |
| P1-02 | MoE YAML contracts not implemented | **FIXED** — z-loss + aux-loss-free bias; capacity still open (P2) |
| P1-03 | `moe:` config section ignored by loader | **FIXED** — `load_model_config` reads `moe:` |
| P1-04 | Production tokenizer (131k) does not exist | OPEN |
| P1-05 | H200 train corpus path missing | **FIXED** — points at golden synthetic records |
| P1-06 | First real gate (35B) cannot tokenize at declared vocab | OPEN (depends on P1-04) |
| P1-07 | Canonical trainer is smoke-named | **FIXED** — `scripts/train_magi.py` |

---

## 4. ARCHITECTURE TRACE

Input: `"The future of intelligence"`

| Stage | FILE | SYMBOL | INPUT | OUTPUT | OWNED BY MAGI? |
|-------|------|--------|-------|--------|----------------|
| text → ids | `magi/tokenizer/byte_bpe.py` | `MagiByteBPETokenizer.encode` | str | `list[int]` | **YES** (smoke vocab) |
| ids → emb | `magi/model/transformer.py` | `MAGITransformer.token_embedding` | `[B,S]` long | `[B,S,D]` | **YES** |
| block | `magi/model/transformer.py` | `TransformerBlock.forward` | hidden | hidden + residual | **YES** |
| attn | `magi/model/layers.py` | `GQAAttention` + RoPE + causal SDPA/flash | normed x | attn out + KV | **YES** |
| dense FFN (early layers) | `magi/model/layers.py` | `SwiGLU` | normed x | FFN | **YES** |
| MoE router | `magi/model/moe.py` | `MoERouter` | flat x | top-k indices + weights | **YES** |
| routed experts | `magi/model/moe.py` | `MoELayer.routed_experts[*]` | masked tokens | weighted sum | **YES** |
| shared experts | `magi/model/moe.py` | `MoELayer.shared_experts[*]` | all tokens | always-on add | **YES** |
| residual stream | `TransformerBlock` | `x = x + ffn(...)` | | | **YES** |
| final norm | `transformer.py` | `final_norm` | | | **YES** |
| LM head | `transformer.py` | tied `E.weight.T` or `lm_head` | hidden | logits `[B,S,V]` | **YES** |
| sampler | historically `transformers.GenerationMixin` via `MagiForCausalLM` | logits → next id | next token | **YES weights / NO native loop** (pre-fix) |

No stage exits to an external LLM.

Primitives present: decoder-only · RMSNorm · RoPE · GQA · SwiGLU · Sparse MoE · shared expert · Top-K · causal mask · from-scratch `init_magi_weights`.

---

## 5. TRAINING STEP TRACE

Canonical executable path: `scripts/t4_train_smoke.py` / `scripts/h200_train.py` → `magi.train.loop.train_steps`.

| Stage | FILE | SYMBOL | OWNED? |
|-------|------|--------|--------|
| corpus lines / jsonl | `magi/train/data.py` | `load_corpus_lines` | YES |
| tokenize + pack | `pack_texts` | BOS/EOS windows | YES |
| batch | `PackedTokenBatch` | ids / mask / labels | YES |
| forward | `MAGITransformer.forward` | logits | YES |
| shift + CE | `magi/train/loss.py` | `causal_lm_loss` | YES |
| backward | `loss.backward()` / AMP scaler | grads | YES |
| clip + AdamW | `train_steps` | `optimizer.step()` | YES |
| checkpoint | `magi/train/checkpoint.py` | `save_train_checkpoint` | YES |

### Gradient reachability (code-level)

| Parameter group | Gradients expected? | Notes |
|-----------------|---------------------|-------|
| embeddings | YES | CE → logits → tied emb or separate |
| attention QKVO | YES | residual path |
| dense SwiGLU (early layers) | YES | |
| router `gate` | YES | soft weights differentiable; discrete top-k index is STE-free discrete (standard MoE) |
| selected routed experts | YES | weighted residual |
| unselected experts | NO this step | expected; collapse risk without balance |
| shared experts | YES | always applied |
| LM head / tied emb | YES | |

### MoE training controls

| Control | YAML | Code |
|---------|------|------|
| sigmoid-normalize top-k | yes | **implemented** |
| aux-loss-free bias | claimed | **was missing → P1 fix** |
| router z-loss | coeff in YAML | **was missing → P1 fix** |
| capacity factor | claimed | **not implemented** (P2) |
| hierarchical routing hooks | claimed | **not implemented** (P2) |
| expert utilization telemetry | — | `collect_router_telemetry` **yes** |

---

## 6. EXTERNAL MODEL DEPENDENCY MAP

| Occurrence | Class | Role |
|------------|-------|------|
| `magi/hf/*` + `AutoModelForCausalLM.register(MagiConfig, MagiForCausalLM)` | ALLOWED compat | Load/save MAGI under HF API |
| `docs/HF_COMPATIBILITY.md` `from_pretrained` | docs | MAGI checkpoint only |
| `tests/test_hf_compatibility.py` | test | MAGI roundtrip |
| OpenAI/Anthropic/Grok/Gemini/… | — | **not present** |
| Synthetic generators | KEEP | Deterministic templates, not LLM APIs |

**Removing all external LLM APIs does not destroy MAGI.** Removing optional `transformers` currently breaks HF generate/smoke generate until native generate lands.

---

## 7. DENSE / SMOKE / TOY CODE MAP

| Asset | Class | Verdict |
|-------|-------|---------|
| `configs/magi_t4_smoke_v0.1.yaml` | TEST INFRASTRUCTURE | KEEP — MoE kernel smoke, not product |
| `configs/magi_t4_train_smoke_v0.1.yaml` | TEST INFRASTRUCTURE | KEEP |
| `scripts/t4_smoke_run.py` / `t4_train_smoke.py` | TEST / bring-up | KEEP; rename identity via `train_magi.py` |
| `tests/test_*tiny*` / `tiny_native_config` | fixtures | KEEP |
| `configs/magi_7b_moe_v0.1.yaml` | REAL MAGI TRAINING (bring-up scale) | KEEP — **canonical H200 MoE** |
| `configs/magi_35b_moe_v0.1.yaml` | REAL MAGI GATE | KEEP — first declared real gate |
| `configs/magi_400b_v0.1.yaml` | scale envelope | KEEP (config-only until cluster) |
| `configs/magi_7b_v0.1.yaml` dense | dense baseline | **QUARANTINE from roadmap** (explicitly baseline-only) |
| `configs/magi_casual_v0.1.yaml` dense 13.8B | separate subsystem | **QUARANTINE** — not Sparse-MoE MAGI spine |
| `casual/**` RESPONSIBILITY stubs | architecture theater / future | QUARANTINE — no executable model |

**No dense ladder recommended as MAGI v0.** Dense exists as labeled baseline / casual envelope only.

---

## 8. PARAMETER ACCOUNTING

Computed via `scripts/param_count.py` (stdlib, topology formulas matching `MoELayer`/`SwiGLU`/`GQA`).

### MAGI-7B-MoE (`configs/magi_7b_moe_v0.1.yaml`) — current H200 bring-up

| Component | Params |
|-----------|--------|
| Embeddings | 16,777,216 |
| LM head (tied) | 0 |
| Attention all layers | 402,653,184 |
| Dense FFN (3 layers) | 103,809,024 |
| MoE all layers | 5,933,498,368 |
| Norms | 133,120 |
| **TOTAL** | **6,456,870,912** |
| **ACTIVE / token** | **966,526,976** (~top_k+shared experts) |

YAML `expected_param_count` **matches**.

Note: active formula excludes embedding table (lookup cost not counted). Report as accounting convention, not silent mismatch with TOTAL.

### MAGI-35B-MoE-A8B

| | |
|--|--|
| TOTAL | 34,054,674,432 |
| ACTIVE/token | 8,335,202,304 |
| YAML match | YES |

### MAGI-400B / CASUAL dense / 7B dense

| Model | Total | Active |
|-------|-------|--------|
| 400B MoE | 406.626B | 43.165B |
| CASUAL dense | 13.789B | 13.789B |
| 7B dense baseline | 6.413B | 6.413B |

---

## 9. TOKENIZER AUDIT

| Capability | Status |
|------------|--------|
| Train Byte-BPE from corpus | YES — `train_byte_bpe` |
| NFKC + UTF-8 byte pieces | YES |
| Specials BOS/EOS/PAD/UNK | YES |
| Serialize / load JSON | YES — `tokenizer/artifacts/magi_t4_smoke_v0.1.json` |
| Smoke seed corpus | YES — `tokenizer/data/t4_smoke_seed.txt` |
| Production 131k train pipeline | **NO** — experiment YAML only |
| Chat template / identity tokens | **NO** |
| Used by train + infer | YES for smoke vocab |

**Owning a language stack:** partial. Smoke tokenizer is MAGI-owned. Production MAGI tokenizer is not shipped.

---

## 10. CHECKPOINT AUDIT

| Item | Status |
|------|--------|
| Format | `magi_single_gpu_v0.2` |
| Weights | `model.safetensors` (canonical) |
| Meta | `train_meta.json` + `CheckpointManifest` |
| Optimizer | optional `optimizer.pt` |
| Resume | `load_train_checkpoint` |
| Init from scratch | `MAGITransformer(cfg)` + `init_magi_weights` — no `from_pretrained` foreign weights |
| Observed artifact | `artifacts/t4_train_smoke/train.pt` (legacy v0.1 blob present) |

---

## 11. GENERATION AUDIT

| Path | Status |
|------|--------|
| HF `MagiForCausalLM.generate` | Works if transformers installed; uses MAGI logits |
| Native HF-free generate | **was missing** → implementation required |
| `magi chat` REPL | **was missing** |
| Identity from weights vs prompt | Untrained / lightly trained smoke → garbage text OK; must remain MAGI weights |

---

## 12. KEEP / REFACTOR / QUARANTINE / DELETE

### KEEP
- `magi/model/{transformer,layers,moe,outputs,torch_runtime}.py`
- `magi/train/{loop,loss,data,checkpoint}.py`
- `magi/tokenizer/byte_bpe.py`
- `magi/config/loader.py`
- `configs/magi_7b_moe_v0.1.yaml`, `magi_35b_moe_v0.1.yaml`, `magi_400b_v0.1.yaml`, T4 smoke configs
- `scripts/param_count.py`, `validate_*`, synthetic data builders
- unit tests with tiny dims

### REFACTOR
- `scripts/h200_train.py` / smoke-named trainer → `scripts/train_magi.py` façade
- Generation: native path primary; HF generate optional
- Load `moe:` into runtime config; implement balance + z-loss
- Point H200 corpus at buildable synthetic path

### QUARANTINE
- `casual/**` stub tree + dense CASUAL 13.8B as non-spine product
- `configs/magi_7b_v0.1.yaml` dense baseline (label already correct — keep out of default train)
- HF Auto registration (compat only; not cognition)

### DELETE
- Nothing mandatory now. Do **not** delete smoke tests.
- Future: remove any path that routes generation to external LLM if introduced.

---

## 13. MINIMUM PATH TO FIRST REAL MAGI CHECKPOINT

```text
1. MAGI-7B-MoE config (Sparse-MoE family member — not dense ladder)
2. Smoke MAGI Byte-BPE (8192) — accept as bring-up tokenizer
3. Synthetic / seed corpus (build if H200 jsonl missing)
4. python scripts/train_magi.py --config configs/magi_7b_moe_v0.1.yaml --profile configs/magi_7b_train_h200_v0.1.yaml
5. Random init → causal CE → AdamW → artifacts/.../model.safetensors
6. New process: load safetensors → native generate / magi_chat
7. Scale gate: train production tokenizer 131k → switch to MAGI-35B-MoE-A8B
```

First **architecture-family** trainable model today: **MAGI-7B-MoE** (topology child of 35B).  
Declared **program first gate**: **MAGI-35B-MoE-A8B** (blocked on tokenizer + cluster memory).

---

## 14. EXACT FILES TO MODIFY

| File | Change |
|------|--------|
| `AUDIT_MAGI_NATIVE_LLM.md` | this audit |
| `magi/runtime/generate.py` | native greedy/sample decode |
| `magi/model/moe.py` | expert bias + z-loss hooks |
| `magi/train/loss.py` | combine CE + router z-loss |
| `magi/train/loop.py` | apply aux-loss-free bias updates; total loss |
| `magi/config/loader.py` | load MoE runtime fields |
| `scripts/train_magi.py` | canonical train entry |
| `scripts/magi_chat.py` | native chat REPL |
| `scripts/magi_generate.py` | one-shot generate |
| `configs/magi_7b_train_h200_v0.1.yaml` | corpus fallback that exists |
| `magi/runtime/__init__.py` / `magi/model/__init__.py` | exports |

---

## 15. ACCEPTANCE TESTS

Do **not** run on this audit host (policy). On H200 / CI with deps:

```text
1. Init MAGI-7B-MoE without downloading foreign LLM
2. Tokenize real text with MagiByteBPETokenizer
3. Forward → logits [B,S,V]
4. causal_lm_loss + backward → non-null grads on emb, attn, router, shared, active experts, head
5. optimizer.step changes parameters
6. save model.safetensors
7. new process load + native generate (no transformers required)
8. magi_chat prints tokens from MAGI weights only
9. Smoke configs remain fixtures; default train profile is MoE
10. Dense 7B / CASUAL never default train target
```

End-to-end proof commands (target machine):

```bash
python scripts/build_synthetic_dataset.py --config configs/synthetic_magi_v0.1.yaml
python scripts/train_magi.py --device cuda --steps 50 --seq 512
python scripts/magi_generate.py --checkpoint artifacts/h200_7b_moe_train/model.safetensors \
  --config configs/magi_7b_moe_v0.1.yaml --prompt "The future of intelligence"
python scripts/magi_chat.py --checkpoint artifacts/h200_7b_moe_train/model.safetensors \
  --config configs/magi_7b_moe_v0.1.yaml
```

---

## CONFIGURATION AUTHORITY

| Source | Authority |
|--------|-----------|
| `configs/magi_*_v0.1.yaml` `architecture:` | **primary** for dims |
| `ModelConfig` | runtime mirror (incomplete for `moe:` historically) |
| `MagiConfig` (HF) | derived via convert — must not diverge |
| CLI `--seq/--steps` | train schedule only; must not silently change architecture |
| Tiny test `ModelConfig(...)` | fixtures only |

---

## POST-AUDIT IMPLEMENTATION (2026-08-11)

| Item | Status |
|------|--------|
| `magi/runtime/generate.py` HF-free decode | **DONE** |
| `scripts/magi_generate.py` / `scripts/magi_chat.py` | **DONE** |
| `scripts/train_magi.py` canonical MoE train entry | **DONE** |
| Router z-loss + aux-loss-free expert bias | **DONE** |
| `moe:` YAML → `ModelConfig` | **DONE** |
| H200 corpus path → existing golden synthetic | **DONE** |
| Restored deleted `magi_casual` / `tokenizer_t4_smoke` configs | **DONE** |
| Production 131k tokenizer | **OPEN (P1-04)** |
| Capacity factor / hierarchical routing | **OPEN (P2)** |
| End-to-end proof on this host | **SKIPPED** (no deps / no local test policy) |

Updated native generation gate: **YES** (code path). Casual chat readiness: **PARTIAL** (REPL exists; production tokenizer + trained checkpoint required).

### Production inference invariant (enforced 2026-08-11)

- `scripts/magi_chat.py` / `scripts/magi_generate.py`: checkpoint **required**; smoke tokenizer/config **forbidden**; no random init.
- Tokenizer resolved: `--tokenizer` → `train_meta.json:tokenizer_path` → config `tokenizer.artifact`.
- Decode uses `GenerateResult.new_token_ids` only (no `split("MAGI:")`).
- Token-aware context budget in chat.
- Smoke lives in `scripts/dev/smoke_*.py` and `scripts/t4_*` only.

