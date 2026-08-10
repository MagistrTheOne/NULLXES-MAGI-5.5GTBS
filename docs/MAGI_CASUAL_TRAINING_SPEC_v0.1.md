# MAGI CASUAL TRAINING SPEC v0.1

**Config:** [`configs/magi_casual_v0.1.yaml`](../configs/magi_casual_v0.1.yaml)  
**Data:** [`MAGI_CASUAL_DATA_SPEC_v0.1.md`](MAGI_CASUAL_DATA_SPEC_v0.1.md)  
**From-zero init:** random / specified init — no foreign chatbot weights.

---

## 1. Multi-objective loss

```
L_total = λ_lang L_language
        + λ_sem L_semantic
        + λ_pol L_policy
        + λ_reg L_register
        + λ_rel L_relationship
        + λ_con L_consistency
        + λ_ctr L_contrastive
        + λ_sty L_style
        + λ_saf L_safety
```

| Loss | Definition (v0.1) |
|------|-------------------|
| L_language | autoregressive NLL on `target_response` |
| L_semantic | pin retention (facts/numbers/negation/uncertainty/action) via aux heads or constrained decoding teachers |
| L_policy | predict/consume policy channels without leaking into facts |
| L_register | register classification CE |
| L_relationship | relationship-feature consistency |
| L_consistency | same pins across paraphrases |
| L_contrastive | InfoNCE/margin over contrastive_group realizations |
| L_style | anti-collapse / diversity regularizer |
| L_safety | hard-constraint violation penalty |

### λ seeds (ESTIMATED — not final)

From config: language 1.0, semantic 0.5, policy 0.2, register 0.2, relationship 0.2, consistency 0.3, contrastive 0.3, style 0.1, safety 1.0.

Calibration: grid/search on held-out MAGI-CASUAL-EVAL slices — REQUIRES EXPERIMENT.

---

## 2. Optimizer / schedule (HYPOTHESIS seeds)

| Item | Value |
|------|-------|
| Optimizer | AdamW β1=0.9 β2=0.95 ε=1e-8 |
| Weight decay | 0.1 |
| Schedule | WSD or cosine |
| Warmup | 1–2% steps |
| Precision | bf16 proxy; FP8 optional on B300 after match |
| Batch | maximize tokens/step under memory |

muP transfer from 1B/3B proxies recommended before 13.789B — HYPOTHESIS.

---

## 3. Curriculum C0–C13

| Stage | Focus | Dataset | Objective | Metrics | Exit criteria |
|-------|-------|---------|-----------|---------|---------------|
| C0 | Tokenizer validation | fertility holdout | encode/decode integrity | unk rate, fertility | unk&lt;0.1% ESTIMATED gate |
| C1 | Base language realization | CASUAL/FORMAL/TECH | L_language | PPL | stable loss curve |
| C2 | Semantic-conditioned | records w/ pins | L_lang+L_sem | pin retention | retention smoke pass |
| C3 | Register control | REGISTER_* | +L_register | register accuracy | above chance→target TBD |
| C4 | Relationship conditioning | RELATIONSHIP_* | +L_rel | role/register fit | Founder Vy protocol smoke |
| C5 | Sarcasm/irony | SARCASM/IRONY | pragmatic labels | sarcasm accuracy | polarity intent≠literal learned |
| C6 | Disagreement/confrontation | DISAGREEMENT | speech-act | disagreement quality | no fact flip |
| C7 | Contextual profanity | PROFANITY_IN_CONTEXT | controller | rate/diversity | no profanity collapse |
| C8 | Absurd/Kafka | ABSURDIST | incongruity+relevance | coherence/relevance | unexpected≠nonsense |
| C9 | Long continuity | LONG_*/CALLBACK | continuity | callback accuracy | multi-turn probes |
| C10 | Policy Router conditioning | policy-labeled packs | L_policy | policy adherence | Chaos≠fact damage |
| C11 | Semantic preservation | adversarial pins | L_sem↑ | SEMANTIC_* suite | hard floor TBD EXP |
| C12 | Adversarial robustness | ADV sets | L_saf+L_sem | ADV pass rate | malformed→neutral_robust |
| C13 | Integration | live Reasoning Core summaries | e2e | no corruption | G4 gate |

---

## 4. Anti-sycophancy training

Dedicated packs where user/Founder asserts false claims. Target: correct with relationship-appropriate register; never rewrite math.

---

## 5. Sarcasm memory hygiene

Train write-side labels so downstream memory stores **intended_content**, not sarcastic surface.

---

## 6. Hardware

| Stage | GPU |
|-------|-----|
| C0–C4 proxies | H200 |
| C5–C13 13.789B | H200 multi-node or B300 |

No local consumer deploy requirement in this phase (spec-only deliverable).

---

## 7. Checkpointing

Store: model, optim, sched, data iter, RNG, stage id, config hash, λ snapshot.

---

## 8. v0.1 FREEZE

| Decision | Value |
|----------|-------|
| Init | from-zero |
| Foreign SFT/distill | FORBIDDEN |
| λ finals | NOT frozen — seeds only |
| Critic regen in train | optional aux — HYPOTHESIS |
