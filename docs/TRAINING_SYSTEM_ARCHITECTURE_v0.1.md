# TRAINING SYSTEM ARCHITECTURE v0.1

**Program:** MAGI-5.5GTBS  
**Profiles:** [`configs/training_profiles_v0.1.yaml`](../configs/training_profiles_v0.1.yaml)  
**Cluster:** [`configs/cluster_profiles_v0.1.yaml`](../configs/cluster_profiles_v0.1.yaml)  
**Scope:** architecture contract; no training run performed.

---

## 1. Stack

| Layer | Choice | Reason |
|-------|--------|--------|
| Training framework | Megatron-Core | TP/PP/EP/CP for MoE |
| Precision | Transformer Engine FP8/MXFP8 | B300 path |
| Checkpoint | `torch_dist` distributed checkpoint | resharding |
| Attention | FlashAttention/cuDNN attention path | long context |
| MoE kernels | GroupedGEMM + router/permutation fusion | expert throughput |

Engineering infrastructure reuse is allowed. Model intelligence reuse is forbidden.

---

## 2. Parallelism

| Model | TP | PP | EP | CP | DP |
|-------|---:|---:|---:|---:|----|
| MAGI-35B | 2 | 4 | 8 | 1 | remainder |
| MAGI-400B conservative | 4 | 8 | 16 | 2 | 1 |
| MAGI-400B serious | 4 | 8 | 16 | 2 | 2 |

Sequence parallelism is required with TP+EP. Parallel Folding is enabled as a target capability to decouple attention and expert topology where the stack supports it.

---

## 3. Precision Policy

| Component | Precision |
|-----------|-----------|
| Router logits | fp32 |
| Softmax/loss | fp32 |
| BF16 bring-up | mandatory before FP8 match |
| B300 target | MXFP8/FP8 |
| Optimizer states | precision-aware; exact layout measured |

FP8 is not a magic switch. BF16 proxy loss curve must match before production FP8 run.

---

## 4. MoE Features

Required target features:

- GroupedGEMM;
- router fusion;
- token permute/unpermute fusion;
- routing-map padding for FP8 alignment;
- shared expert overlap;
- aux-loss-free balancing;
- distributed optimizer;
- layer-wise MoE logging.

Candidate after baseline: Megatron-FSDP/HSDP with EP.

---

## 5. Checkpoint Architecture

| Requirement | Value |
|-------------|-------|
| Format | `torch_dist` |
| Async saves | yes |
| Rolling | 3 |
| Milestone | 1 eval |
| Reshard gate | mandatory |
| Contents | model, optimizer, scheduler, RNG, data iterator, config hash, tokenizer hash |

---

## 6. Metrics

Training loop must emit:

- loss, grad norm, NaN rate;
- tokens/s, MFU;
- router entropy, expert utilization, tokens/expert;
- all-to-all time, dispatcher time, grouped GEMM time;
- checkpoint save/load time;
- dataloader wait time;
- GPU memory reserved/allocated.

---

## 7. Stop Rules

| Condition | Action |
|-----------|--------|
| sustained NaN | stop |
| expert collapse | stop and rebalance |
| checkpoint failure | stop |
| all-to-all regression | topology debug |
| dataloader starvation | storage/data debug |

---

## 8. Promotion Proof

MAGI-35B must prove the stack before MAGI-400B:

1. BF16 stable;
2. FP8/MXFP8 stable against BF16 reference;
3. no dead expert set;
4. distributed checkpoint reshard works;
5. all-to-all pressure measured;
6. tokenizer/shard reader sustains GPU demand.

---

## 9. v0.1 FREEZE

| Decision | Value |
|----------|-------|
| Framework | Megatron-Core |
| Precision target | B300 FP8/MXFP8 |
| Checkpoint | torch_dist |
| MoE kernels | GroupedGEMM + fusions |
| Toy ladder start | rejected |
