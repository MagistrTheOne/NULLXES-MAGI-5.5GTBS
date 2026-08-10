# MAGI-400B HARD ARCHITECTURE SPEC v0.1

**Program parent:** MAGI-5.5GTBS (FINAL MONSTER)  
**Node role:** baseline monster — mandatory gate before ~1T  
**Config SoT:** [`configs/magi_400b_v0.1.yaml`](../configs/magi_400b_v0.1.yaml)  
**Validator:** `python scripts/param_count.py --config configs/magi_400b_v0.1.yaml`  
**From-zero:** YES. No pretrained backbone. No API cognition.

Claim tags: **KNOWN | CALCULATED | ESTIMATED | HYPOTHESIS | REQUIRES EXPERIMENT**

---

## 1. Exact candidate model topology

| Field | Value | Class |
|-------|-------|-------|
| Type | Decoder-only, pre-norm, sparse MoE FFN | CALCULATED design |
| `d_model` | 8192 | CALCULATED |
| `n_layers` | 64 (0–3 dense FFN; 4–63 MoE) | CALCULATED |
| `n_heads` / `n_kv_heads` / `d_head` | 64 / 8 / 128 | CALCULATED |
| Attention | GQA; FlashAttention-compatible; hybrid local/global 3:1 | HYPOTHESIS (hybrid) |
| Local window | 4096 | HYPOTHESIS |
| Positional | RoPE θ=1e6 on local; NoPE on full-attn layers | HYPOTHESIS |
| Norm | RMSNorm ε=1e-6 | KNOWN practice choice |
| Activation | SwiGLU; no bias | KNOWN practice choice |
| `d_ff_dense` | 22016 | CALCULATED |
| `d_ff_expert` | 2048 | CALCULATED |
| Routed / Shared / Top-k | 128 / 2 / 8 | CALCULATED |
| Vocab | 131072 (NULLXES MAGI tokenizer) | HYPOTHESIS size |
| Embeddings | Untied | CALCULATED |
| Train / infer context | 32768 / 131072 | HYPOTHESIS |

**WHY:** Sparse MoE delivers ≥400B capacity at ~43B active — fits MAGI-5.5GTBS active-cycle band.  
**COST:** Expert parallelism + all-to-all.  
**FAILURE PREVENTED:** Dense 400B training memory explosion.  
**REJECTED:** Dense-only 400B; adopting foreign pretrained MoE weights; shrinking below 400B as “the architecture.”

Block:

```
x → RMSNorm → GQA(+RoPE|NoPE) → + → RMSNorm → DenseFFN|MoE(SwiGLU) → +
```

---

## 2. Exact parameter accounting (CALCULATED)

Equations:

```
P_emb     = V · d
P_lm      = V · d                          # untied
P_attn_L  = d² + 2 · d · (n_kv · d_h) + d²
P_dense   = 3 · d · d_ff_dense
P_expert  = 3 · d · d_ff_expert
P_moe_L   = (E_r + E_s) · P_expert + d · E_r
P_norm    = L · 2 · d + d
P_total   = P_emb + P_lm + L·P_attn_L + L_d·P_dense + L_m·P_moe_L + P_norm
```

| Component | Params | Human |
|-----------|-------:|------:|
| Embeddings | 1,073,741,824 | 1.074B |
| LM head | 1,073,741,824 | 1.074B |
| Attention (64×) | 9,663,676,416 | 9.664B |
| Dense FFN (4×) | 2,164,260,864 | 2.164B |
| MoE (60×) | 392,649,768,960 | 392.650B |
| Norms | 1,056,768 | 1.057M |
| **TOTAL** | **406,626,246,656** | **406.626B** |

`TOTAL ≥ 400B` — VALID. Reconciles with `param_count.py`.

---

## 3. Active-parameter accounting (CALCULATED)

```
P_active = L·P_attn_L + L_d·P_dense
         + L_m · ((top_k + E_s)·P_expert + d·E_r)
         + P_lm + P_norm
```

| Metric | Params | Human |
|--------|-------:|------:|
| ACTIVE / token | 43,164,639,232 | **43.165B** |
| Active experts / MoE layer | 10 (8+2) | — |
| Cognitive cycle 1-pass | 43,164,639,232 | 43.165B |
| Cognitive cycle 2-pass | 86,329,278,464 | 86.329B |

Matches MAGI-5.5GTBS envelope 45–90B / cycle (ESTIMATED compatibility).

---

## 4. Attention architecture

| Decision | Choice | WHY | COST | PREVENTS | REJECTED |
|----------|--------|-----|------|----------|----------|
| GQA 8:1 | 64Q / 8KV | KV cache shrink | slight quality risk vs MHA | inference RAM blowup | full MHA |
| Hybrid 3:1 | local 4k + full | long-ctx FLOPs | kernel complexity | quadratic wall | all-full attn |
| RoPE+NoPE | RoPE local, NoPE full | long-range binding hyp. | ablation load | RoPE stretch failure | pure absolute PE |
| FlashAttn-compatible | shapes for FA2/3 | MFU | kernel deps | naive attn OOM | custom MLA-first |

MLA reserved if KV-bound at ≥128k batch — REQUIRES EXPERIMENT.

---

## 5. MoE architecture

| Field | Value |
|-------|------:|
| MoE layers | 60 |
| Dense prefix | 4 |
| Expert FFN | SwiGLU `d_ff_e=2048` |
| Routed experts | 128 |
| Shared experts | 2 always-on |
| Top-k | 8 |
| Capacity factor (train) | 1.0–1.25 ESTIMATED |

**WHY shared experts:** stabilize common computation; reduce dead-expert pressure.  
**REJECTED:** Switch-Transformer top-1 only; no shared expert; MoE from layer 0.

---

## 6. Routing architecture

**v0.1 base (trainable):**

```
logits = W_gate @ x                         # fp32
scores = sigmoid(logits)
scores = scores / sum(scores)
idx, w = top_k(scores, k=8)
y = Σ w_i Expert_i(x) + Σ Shared_j(x)
```

Load balance: aux-loss-free expert bias (SMEBU-style) — HYPOTHESIS at this scale.  
Router z-loss coeff `1e-5` — HYPOTHESIS.

**Extension hooks (not Stage-0 training):**

```
STATE → domain_router → cognitive_router → expert_family_router → specialists
```

Instrumentation (mandatory): `tokens_per_expert`, `router_entropy`, `expert_utilization`, dead-expert count, imbalance ratio.

Failure modes: collapse, dead experts, router gaming, over-specialization, all-to-all congestion.

---

## 7. Tokenizer architecture

| Decision | Value | Class |
|----------|-------|-------|
| Ownership | NULLXES MAGI tokenizer | LAW |
| Vocab | 131072 | HYPOTHESIS |
| Algorithms to evaluate | BPE, Unigram, byte-hybrid | REQUIRES EXPERIMENT |
| Languages | RU, EN, code, math, technical | KNOWN corpus intent |
| Inherit foreign vocab | FORBIDDEN | LAW |

Metrics: tokens/char, tokens/word, compression, code fragmentation, RU morphology, math symbols, rare-symbol rate.  
Corpus pipeline: RAW → provenance → license → parse → normalize → langID → quality → exact/fuzzy dedup → contamination → privacy → domain → curriculum → tokenize → shards.

---

## 8. Numerical precision strategy

| Stage | Precision | Class |
|-------|-----------|-------|
| Proxy (H200) | bf16 matmul + fp32 master | KNOWN stack |
| Production (B300) | FP8 TE after bf16 proxy match | HYPOTHESIS transfer |
| Router / softmax / loss | fp32 always | LAW |
| Grad scaler | TE/framework default | ESTIMATED |

---

## 9. Distributed training topology

B300 template (ESTIMATED — profile before lock):

| Axis | Value |
|------|------:|
| TP | 4 |
| PP | 8 |
| EP | 16 |
| CP | 2 (ctx>32k) |
| DP | remainder |
| Node | 8×B300 HGX |

Activation checkpointing: selective — ESTIMATED.  
Stack: Megatron-Core + Transformer Engine — infrastructure reuse only.

---

## 10. Accelerator configurations

| Role | GPU | HBM/GPU | Notes |
|------|-----|--------:|-------|
| Proxy / muP | H200 | 141 GB | ≤7B–engineering |
| MAGI-400B pretrain | B300 | 288 GB | FP8 default path |
| Forbidden default | H100/A100 | — | not planned |

Exact node count = REQUIRES EXPERIMENT (memory profiler + batch search).  
Order-of-magnitude ESTIMATED: hundreds of B300 GPUs for serious pretrain (not a measured quote).

---

## 11. Memory requirements

Weights BF16: `406626246656 × 2 ≈ 813.25 GB` — CALCULATED.  
AdamW footprint rule-of-thumb ~14 B/param (bf16 + fp32 master + m + v):  
`406626246656 × 14 ≈ 5.693 TB` — ESTIMATED before sharding/activations.

| Item | Class |
|------|-------|
| Model weights BF16 | CALCULATED |
| Optimizer+master aggregate | ESTIMATED |
| Activations | ESTIMATED (seq, microbatch dependent) |
| KV cache inference | CALCULATED formula: `2·L·n_kv·d_h·seq·batch·bytes` |
| Expert buffers / capacity padding | ESTIMATED |
| FP8 weight store ~half of BF16 | ESTIMATED with TE overhead |

Fitting weights ≠ trainable. Sharding via TP/PP/EP/DP mandatory.

---

## 12. Communication requirements

| Domain | Path | Bottleneck risk |
|--------|------|-----------------|
| Intra-node | NVLink 5 (B300) | TP collectives |
| Inter-node | IB/RoCE 800Gb-class | DP + EP all-to-all |
| MoE | expert all-to-all | **primary training bottleneck** ESTIMATED |

---

## 13. Storage / checkpoint requirements

| Item | Value | Class |
|------|------:|-------|
| BF16 weights ckpt | ~813 GB | CALCULATED |
| Full train state | multi-TB | ESTIMATED |
| Policy | async dist ckpt; 3 rolling + 1 eval milestone | HYPOTHESIS ops |
| Must store | model, optim, sched, data iter, RNG, step, config hash | LAW |

---

## 14. Training-token scenarios

MoE budgets vs **active** params (Chinchilla-style guidance — ESTIMATED):

| Scenario | Tokens | Notes |
|----------|-------:|-------|
| Minimal research | 2T | undertrain risk |
| Baseline | 5T | ESTIMATED |
| Strong | 10T | ESTIMATED |
| Heavy | 15T | data-wall risk |

Data mix seeds (ESTIMATED, tune by eval): general 50–60%, code 15–25%, STEM 10–15%, multilingual 5–10%.

---

## 15. Approximate training FLOPs

Dense rule: `FLOPs ≈ 6 · N · T`.  
Sparse refinement (ESTIMATED):

```
FLOPs ≈ 6 · N_active · T · (1 + α_attn + α_recompute + α_aux)
```

Example (ESTIMATED, α≈0.3):  
`6 × 43.165e9 × 5e12 × 1.3 ≈ 1.69e24 FLOPs` for 5T tokens.

Uncomfortable numbers are intentional. Redesign only via ladder experiments, not by deleting 400B.

---

## 16. Likely infrastructure bottlenecks

1. MoE all-to-all (EP)  
2. Checkpoint I/O bandwidth  
3. Router imbalance → stragglers  
4. PP bubbles at depth 64  
5. Long-context CP communication  

---

## 17. Catastrophic architectural failure modes

| Mode | Detection | Mitigation |
|------|-----------|------------|
| Expert collapse | util histogram | bias updates, capacity, dropout-of-experts research |
| Dead experts | <1% util persistent | reinit / aux pressure |
| Loss NaN | fp32 checks | reduce LR, disable FP8 temporarily |
| OOM | allocator traces | cut microbatch, raise PP/EP |
| Router gaming | entropy collapse | z-loss, noise |
| Semantic-unrelated routing | probe tasks | hierarchical hooks later |

---

## 18. Validation models required before 400B run

| Vehicle | Purpose | Is MAGI? |
|---------|---------|----------|
| MAGI-Sim | kernels | NO |
| ~1B dense | init/muP | NO |
| ~7B dense | stack | NO |
| 30–70B MoE | routing/EP/FP8 | NO |
| **MAGI-400B** | baseline monster | **YES** |

---

## 19. Experiments that could falsify the architecture

1. Aux-loss-free routing unstable at 60 MoE layers → revisit aux loss.  
2. Hybrid NoPE full layers hurt needle/long-ctx → revert RoPE-full.  
3. Active 43B insufficient vs dense A35B-class proxies on reasoning → raise top_k / d_ff_e.  
4. EP all-to-all MFU unusable on available fabric → reduce E_r or change parallel map.  
5. Tokenizer fertility fails RU/code → vocab redesign (not foreign inheritance).

---

## 20. Migration path MAGI-400B → MAGI-5.5GTBS

```
MAGI-400B (406.626B / 43.165B active)
  → grow routed expert mesh + hierarchical routers (~1T)
  → attach memory/world/self interfaces (multi-T)
  → allocate measured subsystem budgets → MAGI-5.5GTBS (5.5T / 45–90B active cycle)
```

Preserve: GQA/SwiGLU/RMSNorm/shared+routed MoE pattern, tokenizer family, from-zero law, Policy Router + CASUAL outside reasoning weights.

See [`MAGI_5.5GTBS_SCALE_PATH_v0.1.md`](MAGI_5.5GTBS_SCALE_PATH_v0.1.md).

---

## Decision ledger (HARD v0.1 FREEZE)

| Decision | Value |
|----------|-------|
| Program parent | MAGI-5.5GTBS |
| TOTAL | 406,626,246,656 |
| ACTIVE/token | 43,164,639,232 |
| Pretrained weights | FORBIDDEN |
| GPU policy | H200 proxy / B300 production |
| Personality in HARD | OUT OF SCOPE |

Personality / CASUAL / Policy Router are separate specs.
