# MAGI-35B-MoE-A8B ARCHITECTURE SPEC v0.1

**Program parent:** MAGI-5.5GTBS  
**Role:** first real trainable gate, replacing toy-model ladder start  
**Config SoT:** [`configs/magi_35b_moe_v0.1.yaml`](../configs/magi_35b_moe_v0.1.yaml)  
**Validator:** `python scripts/param_count.py --config configs/magi_35b_moe_v0.1.yaml`  
**From-zero:** YES. No pretrained backbone. No API cognition.

Claim tags: **KNOWN | CALCULATED | ESTIMATED | HYPOTHESIS | REQUIRES EXPERIMENT**

---

## 1. Position In MAGI-5.5GTBS Ladder

MAGI-35B-MoE-A8B is the first trainable gate that exercises real MoE routing, expert parallelism, FP8 path, tokenizer pressure, and cluster all-to-all behavior.

```mermaid
flowchart LR
  KernelSmoke[KernelSmokeFixtures] --> M35[MAGI_35B_MoE_A8B]
  M35 --> M400[MAGI_400B]
  M400 --> M1T[MAGI_1T]
  M1T --> M55[MAGI_5.5GTBS]
```

Kernel smoke fixtures may exist for unit tests, but they are not model milestones.

---

## 2. Exact Topology

| Field | Value | Class |
|-------|-------|-------|
| Type | Decoder-only, pre-norm, sparse MoE | CALCULATED design |
| `d_model` | 6144 | CALCULATED |
| `n_layers` | 48 | CALCULATED |
| Dense prefix / MoE body | 4 / 44 | CALCULATED |
| `n_heads` / `n_kv_heads` / `d_head` | 48 / 8 / 128 | CALCULATED |
| Attention | GQA causal | CALCULATED |
| Positional | RoPE θ=1e6 | HYPOTHESIS |
| Norm | RMSNorm ε=1e-6 | KNOWN practice choice |
| Activation | SwiGLU | KNOWN practice choice |
| `d_ff_dense` | 16384 | CALCULATED |
| `d_ff_expert` | 512 | CALCULATED |
| Routed / shared / top-k | 64 / 1 / 4 | CALCULATED |
| Vocab | 131072 shared MAGI | HYPOTHESIS |
| Embeddings | Untied | CALCULATED |
| Train / infer context | 32768 / 65536 | HYPOTHESIS |

**WHY:** enough total parameters to validate MoE system behavior without burning 400B-scale allocation immediately.  
**COST:** real EP all-to-all, FP8 kernel requirements, checkpoint pressure.  
**FAILURE PREVENTED:** wasting B300 cluster cycles on toy 1B/7B ladder.  
**REJECTED:** 1B/7B training as project start; dense-only 30B; pretrained model adaptation.

---

## 3. Parameter Accounting

Equations:

```
P_emb     = V · d
P_lm      = V · d
P_attn_L  = d² + 2 · d · (n_kv · d_h) + d²
P_dense   = 3 · d · d_ff_dense
P_expert  = 3 · d · d_ff_expert
P_moe_L   = (E_r + E_s) · P_expert + d · E_r
P_norm    = L · 2 · d + d
P_total   = P_emb + P_lm + L·P_attn_L + L_d·P_dense + L_m·P_moe_L + P_norm
```

| Component | Params | Human |
|-----------|-------:|------:|
| Embeddings | 805,306,368 | 805.306M |
| LM head | 805,306,368 | 805.306M |
| Attention 48× | 4,227,858,432 | 4.228B |
| Dense FFN 4× | 1,207,959,552 | 1.208B |
| MoE 44× | 27,007,647,744 | 27.008B |
| Norms | 595,968 | 0.596M |
| **TOTAL** | **34,054,674,432** | **34.055B** |

---

## 4. Active Parameter Accounting

```
P_active = L·P_attn_L + L_d·P_dense
         + L_m · ((top_k + E_s)·P_expert + d·E_r)
         + P_lm + P_norm
```

| Metric | Params | Human |
|--------|-------:|------:|
| ACTIVE / token | 8,335,202,304 | **8.335B** |
| Active experts / MoE layer | 5 (4+1) | — |
| Cognitive cycle 1-pass | 8,335,202,304 | 8.335B |
| Cognitive cycle 2-pass | 16,670,404,608 | 16.670B |

This is not the final active-cycle band. It is the first serious MoE gate before MAGI-400B.

---

## 5. Training Gate Purpose

MAGI-35B-MoE-A8B must prove:

- tokenizer fertility under 32k context;
- MoE dispatcher correctness;
- aux-loss-free bias viability;
- expert utilization stability;
- FP8/MXFP8 loss-curve viability;
- distributed checkpoint save/load/reshard;
- NCCL all-to-all performance on the target fabric;
- route metrics needed before MAGI-400B.

---

## 6. Parallelism Templates

| Profile | TP | PP | EP | CP | DP | Class |
|---------|---:|---:|---:|---:|----|-------|
| B300 preferred | 2 | 4 | 8 | 1 | remainder | ESTIMATED |
| H200 fallback | 4 | 4 | 8 | 1 | remainder | ESTIMATED |

B300 is preferred because 35B is already a real MoE systems test; H200 fallback exists for constrained burn-in only.

---

## 7. Required Metrics

| Metric | Gate |
|--------|------|
| `tokens_per_expert` | no persistent dead experts |
| `expert_utilization` | no collapse to small expert subset |
| `router_entropy` | stable after warmup |
| `all_to_all_ms` | does not dominate step time permanently |
| `checkpoint_save_s` | within ops budget |
| `loss_nan_rate` | zero sustained NaN |
| `mfu` | measured, no fake target |

---

## 8. Promotion To MAGI-400B

Promotion requires:

1. param reconcile pass;
2. tokenizer gate pass;
3. MoE stability gate pass;
4. checkpoint reshard gate pass;
5. loss curve comparable across bf16 and FP8/MXFP8 profile;
6. no catastrophic expert collapse.

---

## 9. v0.1 FREEZE

| Decision | Value |
|----------|-------|
| First trainable gate | MAGI-35B-MoE-A8B |
| Total params | 34,054,674,432 |
| Active/token | 8,335,202,304 |
| 1B/7B status | kernel smoke only |
| Program parent | MAGI-5.5GTBS |
| Next gate | MAGI-400B |
