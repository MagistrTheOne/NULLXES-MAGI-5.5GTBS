# CLUSTER LAUNCH ARCHITECTURE v0.1

**Program:** MAGI-5.5GTBS  
**Profiles:** [`configs/cluster_profiles_v0.1.yaml`](../configs/cluster_profiles_v0.1.yaml)  
**Scope:** architecture for cluster launch; no deployment performed here.

---

## 1. Hardware Law

| Workload | Primary | Fallback |
|----------|---------|----------|
| MAGI-35B-MoE-A8B | B300 SXM | H200 SXM burn-in only |
| MAGI-400B | B300 SXM | no production fallback |
| CASUAL 13.789B | B300/H200 | B300 for production |
| Serving | B300 preferred | H200 for lower scale |

B300 node assumptions: 8 GPUs, 288GB/GPU, ~2304GB/node HBM, NVLink5/NVSwitch5, 800Gb/s east-west NIC class per GPU.

---

## 2. Training Profiles

| Model | Profile | Nodes | GPUs | TP | PP | EP | CP | DP | Precision |
|-------|---------|------:|-----:|---:|---:|---:|---:|---:|-----------|
| MAGI-35B | B300 minimum | 8 | 64 | 2 | 4 | 8 | 1 | remainder | FP8/MXFP8 |
| MAGI-35B | B300 recommended | 16 | 128 | 2 | 4 | 8 | 1 | remainder | FP8/MXFP8 |
| MAGI-35B | H200 fallback | 16 | 128 | 4 | 4 | 8 | 1 | remainder | bf16→FP8 |
| MAGI-400B | B300 conservative | 128 | 1024 | 4 | 8 | 16 | 2 | 1 | FP8/MXFP8 |
| MAGI-400B | B300 serious | 256 | 2048 | 4 | 8 | 16 | 2 | 2 | FP8/MXFP8 |

These are planning profiles. Final launch uses Megatron memory profiler and NCCL topology measurements.

---

## 3. Network Burn-In

Required before any training run:

1. node inventory and GPU ECC check;
2. per-node NVLink/NVSwitch health;
3. inter-node NCCL all-reduce;
4. inter-node all-to-all at MoE message sizes;
5. checkpoint write/read pressure test;
6. automated bad-node quarantine.

Failure to pass burn-in blocks training.

---

## 4. Storage And Checkpointing

| Layer | Requirement |
|-------|-------------|
| NVMe burst | checkpoint staging and dataloader local cache |
| Parallel filesystem | live distributed checkpoint target |
| Object/archive | long-term milestones |
| Format | Megatron `torch_dist` |
| Policy | async, 3 rolling + 1 eval milestone |
| Reshard | required before promotion gates |

Checkpoint gate must prove save, load, and reshard under a different TP/PP/EP/DP profile.

---

## 5. Serving Profiles

### vLLM EP Path

- enable Expert Parallel;
- DP Attention for KV partitioning where supported;
- FP8 KV cache for long-context pressure;
- prefill/decode split for high-throughput runtime.

### SGLang EP Path

- enable MoE EP;
- use RadixAttention for multi-turn prefix-heavy traffic;
- prefill/decode disaggregation profile;
- structured output path for CASUAL metadata in dev mode.

---

## 6. Required Cluster Telemetry

| Metric | Reason |
|--------|--------|
| step_time_ms | global throughput |
| mfu | no fake utilization claims |
| all_to_all_ms | MoE bottleneck |
| router_entropy | collapse detection |
| tokens_per_expert | load balance |
| checkpoint_save_s | ops budget |
| dataloader_wait_ms | storage bottleneck |
| gpu_memory_reserved | OOM prevention |
| bad_node_count | cluster hygiene |

---

## 7. Launch Sequence

```
cluster_burnin
→ tokenizer_gate
→ shard_read_gate
→ param_reconcile
→ dry_forward
→ dry_backward
→ 100_step_loss_smoke
→ checkpoint_save_load_reshard
→ production_run_window
```

No production run starts before this sequence passes.

---

## 8. v0.1 FREEZE

| Decision | Value |
|----------|-------|
| First trainable cluster model | MAGI-35B-MoE-A8B |
| 400B conservative floor | 1024 B300 GPUs |
| 400B serious baseline | 2048 B300 GPUs |
| Checkpoint format | torch_dist |
| Burn-in gates | mandatory |
