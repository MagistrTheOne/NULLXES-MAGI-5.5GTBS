# MAGI-5.5GTBS Architecture Spec Index

**Final system:** MAGI-5.5GTBS  
**Target envelope:** 5.5T total parameters, 45–90B active parameters per cognitive cycle  
**First real trainable gate:** MAGI-35B-MoE-A8B — 34.055B total / 8.335B active  
**Baseline monster:** MAGI-400B — 406.626B total / 43.165B active  
**Language realization:** MAGI-CASUAL — 13.789B dense  
**Law:** from-zero training, no pretrained backbone, no external LLM API cognition.

---

## Architecture Stack

| Layer | Document | Purpose |
|-------|----------|---------|
| Program ladder | [MAGI_5.5GTBS_SCALE_PATH_v0.1.md](MAGI_5.5GTBS_SCALE_PATH_v0.1.md) | 35B → 400B → 1T → 5.5GTBS progression |
| Master envelope | [MAGI_5.5GTBS_MASTER_ARCHITECTURE_SPEC_v0.2.md](MAGI_5.5GTBS_MASTER_ARCHITECTURE_SPEC_v0.2.md) | 5.5T subsystem decomposition, routing, memory, recurrent cognition |
| First trainable gate | [MAGI-35B_MOE_ARCHITECTURE_SPEC_v0.1.md](MAGI-35B_MOE_ARCHITECTURE_SPEC_v0.1.md) | Real 30–40B MoE gate, no toy-model ladder |
| Baseline monster | [MAGI-400B_HARD_ARCHITECTURE_SPEC_v0.1.md](MAGI-400B_HARD_ARCHITECTURE_SPEC_v0.1.md) | 400B sparse MoE topology and parameter accounting |
| Policy Router | [MAGI_POLICY_ROUTER_SPEC_v0.1.md](MAGI_POLICY_ROUTER_SPEC_v0.1.md) | Founder/Chaos/Research/Corporate/HR control state |
| CASUAL LLM | [MAGI_CASUAL_LLM_ARCHITECTURE_SPEC_v0.1.md](MAGI_CASUAL_LLM_ARCHITECTURE_SPEC_v0.1.md) | Social strategy and language realization |

---

## Production Path

| Area | Document | Purpose |
|------|----------|---------|
| Cluster launch | [CLUSTER_LAUNCH_ARCHITECTURE_v0.1.md](CLUSTER_LAUNCH_ARCHITECTURE_v0.1.md) | B300/H200 profiles, NCCL/all-to-all, checkpoint burn-in |
| Training system | [TRAINING_SYSTEM_ARCHITECTURE_v0.1.md](TRAINING_SYSTEM_ARCHITECTURE_v0.1.md) | Megatron-Core, Transformer Engine, FP8/MXFP8, MoE fusions |
| Inference serving | [INFERENCE_SERVING_ARCHITECTURE_v0.1.md](INFERENCE_SERVING_ARCHITECTURE_v0.1.md) | vLLM/SGLang EP, DP Attention, FP8 KV, prefill/decode split |
| CASUAL production | [CASUAL_PRODUCTION_INTEGRATION_v0.1.md](CASUAL_PRODUCTION_INTEGRATION_v0.1.md) | ReasoningCore → PolicyState → CASUAL → Critic |
| Observability gates | [OBSERVABILITY_AND_GATES_v0.1.md](OBSERVABILITY_AND_GATES_v0.1.md) | Param, tokenizer, cluster, MoE, checkpoint, serving gates |

---

## Data And Tokenization

| Area | Document | Purpose |
|------|----------|---------|
| Tokenizer | [TOKENIZER_ARCHITECTURE_SPEC_v0.1.md](TOKENIZER_ARCHITECTURE_SPEC_v0.1.md) | NULLXES-owned 131072 tokenizer experiment matrix |
| Corpus pipeline | [CORPUS_PIPELINE_ARCHITECTURE_v0.1.md](CORPUS_PIPELINE_ARCHITECTURE_v0.1.md) | RAW→provenance→license→dedup→tokenize→packed shards |
| CASUAL data | [MAGI_CASUAL_DATA_SPEC_v0.1.md](MAGI_CASUAL_DATA_SPEC_v0.1.md) | Contrastive social/pragmatic training records |
| CASUAL training | [MAGI_CASUAL_TRAINING_SPEC_v0.1.md](MAGI_CASUAL_TRAINING_SPEC_v0.1.md) | C0–C13 curriculum and multi-objective loss |
| CASUAL eval | [MAGI_CASUAL_EVAL_SPEC_v0.1.md](MAGI_CASUAL_EVAL_SPEC_v0.1.md) | MAGI-CASUAL-EVAL benchmark contract |
| CASUAL runtime | [MAGI_CASUAL_RUNTIME_SPEC_v0.1.md](MAGI_CASUAL_RUNTIME_SPEC_v0.1.md) | Runtime state, critic, opener/register policy |

---

## Source Of Truth Configs

| Config | Role |
|--------|------|
| [../configs/magi_35b_moe_v0.1.yaml](../configs/magi_35b_moe_v0.1.yaml) | MAGI-35B-MoE-A8B topology |
| [../configs/magi_400b_v0.1.yaml](../configs/magi_400b_v0.1.yaml) | MAGI-400B topology |
| [../configs/magi_casual_v0.1.yaml](../configs/magi_casual_v0.1.yaml) | MAGI-CASUAL topology |
| [../configs/magi_5_5gtbs_architecture_envelope_v0.2.yaml](../configs/magi_5_5gtbs_architecture_envelope_v0.2.yaml) | Final 5.5GTBS envelope |
| [../configs/cluster_profiles_v0.1.yaml](../configs/cluster_profiles_v0.1.yaml) | Cluster profiles |
| [../configs/training_profiles_v0.1.yaml](../configs/training_profiles_v0.1.yaml) | Training profiles |
| [../configs/serving_profiles_v0.1.yaml](../configs/serving_profiles_v0.1.yaml) | Inference profiles |

---

## Validation

```bash
python scripts/param_count.py --all
python scripts/spec_inventory.py
python -m unittest discover -s tests -v
```

No dependency install, no local training, no cluster deployment in this architecture phase.
