# MAGI-5.5GTBS SCALE PATH v0.1

**Program:** MAGI-5.5GTBS  
**Status:** FINAL MONSTER envelope  
**Config:** [`configs/magi_5_5gtbs_envelope_v0.1.yaml`](../configs/magi_5_5gtbs_envelope_v0.1.yaml)  
**From-zero:** YES — no pretrained backbone, no API cognition.

---

## 1. Mission

MAGI-5.5GTBS is the **final monster** of the NULLXES MAGI program:

| Field | Value | Class |
|-------|------:|-------|
| Total parameters | **5.5T** | HYPOTHESIS target |
| Active / cognitive cycle | **45–90B** | HYPOTHESIS target |
| Architecture class | Hierarchical sparse MoE + recurrent cognition + memory/world/self/executive | HYPOTHESIS |

MAGI-400B is a **mandatory baseline node** on the path to 5.5GTBS. It is not the program endpoint.

---

## 2. Scale ladder (authoritative membership)

| Order | Node | Class | Role |
|------:|------|-------|------|
| 0 | MAGI-Sim / ≤1B | validation | kernels, init, muP |
| 1 | Val 1B dense | validation | blocks, tokenizer, data |
| 2 | Val 7B dense | validation | dense stability |
| 3 | Eng 30–70B MoE | engineering | routing, EP, FP8 |
| 4 | **MAGI-400B** | **baseline monster** | **≥400B total / ~43B active — GATE before 1T+** |
| 5 | MAGI-~1T | scale | expert-mesh growth |
| 6 | Multi-trillion | scale | hierarchical routing + memory/world hooks |
| 7 | **MAGI-5.5GTBS** | **FINAL MONSTER** | 5.5T system decomposition |

Rules:

1. Smaller nodes validate assumptions; they do not redefine the target.
2. **MAGI-400B must remain in this list.** Skipping it is invalid.
3. Subsystem budget shares for 5.5T are **not frozen** until 400B (+1T) experiments report.
4. CASUAL (~13B realization) rides the ladder as a separate subsystem; it is **outside** the 5.5T reasoning weight budget.

---

## 3. Subsystem envelope (5.5T) — HYPOTHESIS only

| Subsystem | Share (HYPOTHESIS) | Notes |
|-----------|-------------------:|-------|
| Cognitive cortex | 0.28 | decoder MoE trunk |
| Specialist expert mesh | 0.42 | hierarchical routed capacity |
| World model | 0.08 | predictive state transitions |
| Social / ToM | 0.04 | agent modeling |
| Self model | 0.03 | computational self-state |
| Counterfactual | 0.03 | alternate trajectories |
| Multimodal | 0.06 | later embodiment |
| Executive control | 0.04 | arbitration / compute budget |
| Synthetic drives | 0.02 | bounded control variables |

**Rejected:** freezing these shares before MAGI-400B routing/specialization measurements.

---

## 4. Active compute continuity

| Node | Active / token or cycle | Class |
|------|------------------------:|-------|
| MAGI-400B | ~43B / token; ~43–86B / 1–2 pass cycle | CALCULATED (see param_count) |
| MAGI-5.5GTBS | 45–90B / cognitive cycle | HYPOTHESIS |

Design intent: grow **total** capacity via sparse experts while keeping **active cycle compute** in the same band.

---

## 5. Gates

### Before MAGI-1T

- MAGI-400B TOTAL/ACTIVE reconcile pass (`scripts/param_count.py`)
- Routing stability (no expert collapse)
- EP/FP8 training path proven on B300

### Before MAGI-5.5GTBS

- Measured allocation from 400B and ~1T
- Recurrent cognition hooks validated
- Memory / world / self interfaces stable
- Policy Router + CASUAL integrated without semantic corruption

---

## 6. Hardware policy

| Workload | GPU |
|----------|-----|
| Proxy / ablation | H200 |
| ≥30B / 400B / multi-T pretrain | B300 |

Forbidden as production default: H100, A100, consumer GPUs.

---

## 7. Relationship to Phase 0–1 artifacts

| Artifact | Role |
|----------|------|
| `MAGI-400B_HARD_ARCHITECTURE_SPEC_v0.1.md` | Baseline monster topology |
| `MAGI_POLICY_ROUTER_SPEC_v0.1.md` | Executive social-strategy control |
| `MAGI_CASUAL_*` specs | Realization subsystem |
| This document | Program parent + ladder law |

---

## 8. Known unknowns

| Item | Class |
|------|-------|
| Exact 5.5T subsystem shares | HYPOTHESIS |
| Optimal expert family hierarchy depth | REQUIRES EXPERIMENT |
| Recurrent cycle compute scheduler | REQUIRES EXPERIMENT |
| World/self parameter split | REQUIRES EXPERIMENT |

---

## 9. v0.1 FREEZE

| Decision | Value |
|----------|-------|
| Final monster name | MAGI-5.5GTBS |
| Final total target | 5.5T |
| Active/cycle target | 45–90B |
| MAGI-400B in ladder | MANDATORY |
| From-zero law | MANDATORY |
| CASUAL outside 5.5T reasoning budget | MANDATORY |
