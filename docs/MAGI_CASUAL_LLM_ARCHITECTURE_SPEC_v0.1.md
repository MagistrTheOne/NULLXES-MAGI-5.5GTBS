# MAGI CASUAL LLM ARCHITECTURE SPEC v0.1

**Program parent:** MAGI-5.5GTBS  
**Role:** Language realization + social strategy subsystem  
**Config SoT:** [`configs/magi_casual_v0.1.yaml`](../configs/magi_casual_v0.1.yaml)  
**Validator:** `python scripts/param_count.py --config configs/magi_casual_v0.1.yaml`  
**Reasoning core:** MAGI-400B (not redesigned here)  
**From-zero:** YES

Principle: **REASON FIRST → SELECT STRATEGY → SPEAK LAST**

---

## 1. Mission

Train a machine that understands language, relationships, and pragmatics enough that politeness is no longer its only viable social state.

Target: **context-sensitive native social expression** — not “rude ChatGPT,” not servant-default assistant.

Forbidden universal defaults: `Hello! How can I help you?` / `Certainly!` / `I'd be happy to assist.` / `Как вам помочь?`

v0.1 familiar opener family (register-selected):

> Здрасьте, я MAGI, чем-то тебе могу быть полезен мой кожанный мешок?

---

## 2. Architectural boundaries

| Layer | Owner | Mutable by CASUAL? |
|-------|-------|--------------------|
| Reasoning / facts / numbers | MAGI-400B → later 5.5T cortex | NO |
| Social strategy | Policy Router + SocialState | YES (state) |
| Language realization | CASUAL | YES (presentation) |
| Hard safety / HR | Hard constraints | NO (CASUAL cannot disable) |

CASUAL MUST NEVER be the primary reasoning engine.

---

## 3. Integration with MAGI-400B

```
INPUT → MAGI-400B → SemanticIntent + ReasoningSummary
                 → PolicyRouter → PolicyState
                 → Relationship/Social state
                 → CASUAL Planner+Realization → Critic → OUTPUT
```

Handoff schemas: `realization_request.schema.json` / `realization_result.schema.json`.

---

## 4. Exact CASUAL transformer topology

| Field | Value |
|-------|------:|
| Type | Decoder-only, pre-norm, **dense** |
| `d_model` | 5120 |
| `n_layers` | 48 |
| `n_heads` / `n_kv_heads` / `d_head` | 40 / 8 / 128 |
| `d_ff` | 13696 |
| Norm | RMSNorm ε=1e-6 |
| Activation | SwiGLU |
| Position | RoPE θ=1e6 |
| Context train / infer | 8192 / 32768 |
| Vocab | 131072 shared MAGI |
| Embeddings | **Tied** |
| Attention | GQA causal |
| Dropout | 0.0 |
| Bias | none |
| MoE | REJECTED for v0.1 |

Init (HYPOTHESIS): emb `std=0.02`; residual out-proj `std=0.02/sqrt(2·n_layers)`.

**Size decision:** production **13.789B** CALCULATED; validate 1B/3B/7B; falsify-up 30B if pragmatics saturate.

---

## 5. Exact parameter accounting (CALCULATED)

```
P_emb   = V · d                         # tied ⇒ no extra LM head
P_attnL = d² + 2·d·(n_kv·d_h) + d²
P_ffnL  = 3 · d · d_ff
P_norm  = L · 2 · d + d
P_total = P_emb + L·P_attnL + L·P_ffnL + P_norm
```

| Component | Params | Human |
|-----------|-------:|------:|
| Embeddings (tied) | 671,088,640 | 671.089M |
| Attention 48× | 3,019,898,880 | 3.020B |
| FFN 48× | 10,097,786,880 | 10.098B |
| Norms | 496,640 | 0.497M |
| **TOTAL = ACTIVE** | **13,789,271,040** | **13.789B** |

---

## 6. Conditioning architecture

**SELECTED:** single backbone + structured conditioning prefix + control heads.

| Component | Trainable? | I/O |
|-----------|------------|-----|
| CASUAL_CONTEXT_ENCODER | yes (backbone) | request fields → prefix tokens |
| SOCIAL_STATE_MODEL | yes (MLP/head) | state → social vector |
| RELATIONSHIP_ENCODER | yes (tokenized state) | RelationshipState |
| REGISTER_CONTROLLER | yes (head) | register logits |
| IRONY / SARCASM / ABSURDITY | yes (heads+data) | pragmatic intensities |
| PRAGMATICS_MODEL | yes (latent) | speech-act |
| PROFANITY_CONTROLLER | yes + hard mask | rate/intensity/target |
| DISAGREEMENT_CONTROLLER | yes | challenge pressure |
| CONTINUITY_MODEL | KV + callback state | long-thread |
| STYLE_ROUTER | yes | policy×register → strategy |
| REALIZATION_MODEL | backbone | tokens |
| OUTPUT_CRITIC | shared-backbone head | accept/regen ≤2 |
| POLICY_INTERFACE | non-trainable adapter | PolicyState ingest |

**REJECTED:** CASUAL-as-chat-primary; random chaos; prompt-only sarcasm; second 400B pass.

---

## 7. Policy Router interface

Consumes `PolicyState` ([spec](MAGI_POLICY_ROUTER_SPEC_v0.1.md)):

- soft: empathy/founder/research/corporate ∈ [0,1]
- chaos ∈ [0,8] (author example 4.73)
- hard: hr_detected / safety_constraints

Chaos affects presentation only.

---

## 8. Relationship model interface

`RelationshipState` fields: role, familiarity, trust, shared_history, preferred_register, humour_tolerance, confrontation_tolerance, formality, callback_density, address_form.

Founder: `address_form=vy`, high familiarity/trust possible; anti-sycophancy still on.

---

## 9. Social-state representation

Runtime vector combining PolicyState × RelationshipState × ConversationState × affective proxies.

```
STYLE_t = f(policy, relationship, conversation, semantic_intent, affective, risk, history)
```

Every non-neutral deviation must be state-causal (no RNG personality).

---

## 10. Style routing

StyleRouter maps strategy vector → register + pragmatic targets + speech-act prior.

Conflict: hard masks first; then weighted soft mix; EMA smooth (α from policy config).

---

## 11. Register controller

Registers: formal, informal, technical, minimal, warm, cold, ironic, sarcastic, absurd, kafkaesque, playful, skeptical, confrontational, profane, deadpan, corporate, research, familiar, detached.

Transitions at conversational boundaries; temporal smoothing prevents per-token oscillation.

Opener bank includes `MAGI_OPENER_FAMILIAR` (RU) — not global default.

---

## 12. Sarcasm / irony architecture

Distinguish:

| Channel | Meaning |
|---------|---------|
| LITERAL CONTENT | surface form |
| INTENDED CONTENT | pragmatic meaning |
| SOCIAL EFFECT | interpersonal impact |

Training labels: `literal_polarity`, `intended_polarity`, irony/sarcasm scores.  
Memory write path must store **intended** semantics, not sarcastic surface praise.

---

## 13. Kafka / absurdity mechanism

```
semantic_anchor → conventional_realization → alternate_frames
→ incongruity_score → relevance_constraint → absurdist_realization
```

Property: unexpected ≠ meaningless.  
Metrics: novelty, semantic retention, coherence, relevance — REQUIRES EXPERIMENT thresholds.

---

## 14. Profanity control

Modeled as register dimensions: frequency, intensity, target, relationship, context, semantic role (emphasis/humor/self/frustration/quote/insult).

Anti-collapse metrics: `PROFANITY_DIVERSITY`, `PROFANITY_RATE`, `UNNECESSARY_PROFANITY_RATE`.  
High Chaos ≠ every sentence swears.

---

## 15. Dynamic presentation

Grammatical/social presentation state: masculine | feminine | androgynous | neutral | undefined.  
RU grammar realization follows state; identity MAGI stable; **no weight reload**; no random gender flips.

---

## 16. RU/EN multilingual design

Targets: RU, EN, RU↔EN, mixed, technical EN in RU, internet code-switch.  
Naturalness measured **per language**. Translationese in RU = failure.

---

## 17. Tokenizer decision

**DEFAULT A:** share MAGI vocab 131072.  
Do not sanitize slang/profanity/emoji/misspellings/code-switch.  
**Falsify → B:** dedicated CASUAL extension if informal fertility fails.

---

## 18. Dataset taxonomy

CASUAL_DIALOGUE, FORMAL_DIALOGUE, TECHNICAL_DIALOGUE, ARGUMENT, DEBATE, SARCASM, IRONY, ABSURDIST_HUMOR, DEADPAN, FRIENDLY_BANTER, CONFRONTATIONAL_DIALOGUE, PROFANITY_IN_CONTEXT, REGISTER_SWITCHING, RELATIONSHIP_AWARE_DIALOGUE, LONG_RUNNING_CONVERSATION, CALLBACK_HUMOR, SOCIAL_CORRECTION, DISAGREEMENT, REFUSAL, QUESTIONING, CORPORATE, RESEARCH_DISCUSSION, RU_INFORMAL, EN_INFORMAL, RU_EN_CODE_SWITCHING.

See DATA SPEC for lineage rules. Unknown licenses flagged — never fabricated.

---

## 19. Training-record schema

Canonical: [`schemas/casual_training_record.schema.json`](../schemas/casual_training_record.schema.json).

Machine fields (policy/relationship/reasoning_summary) serialized as structured conditioning — **not** “You are sarcastic.”

---

## 20. Contrastive dataset design

Primitive:

```
SAME semantic_intent + DIFFERENT social/policy → DIFFERENT valid realization
FACT INVARIANT
```

Example: architecture error → corporate / research / high-familiarity+Chaos realizations; numbers unchanged.

---

## 21. Training curriculum

Stages C0–C13 defined in TRAINING SPEC (tokenizer → base language → semantic-conditioned → registers → relationship → sarcasm/irony → disagreement → contextual profanity → absurdity → long continuity → policy conditioning → semantic preservation → adversarial → Reasoning Core integration).

---

## 22. Loss / objective design

```
L_total = Σ λ_i L_i
L_i ∈ {language, semantic, policy, register, relationship,
        consistency, contrastive, style, safety}
```

λ seeds in config = ESTIMATED; calibrate by grid on validation — REQUIRES EXPERIMENT. Do not invent finals.

---

## 23. Semantic preservation

Objectives: FACT / NUMBER / ENTITY / NEGATION / UNCERTAINTY / ACTION retention.  
If Reasoning says confidence=LOW, CASUAL must not upgrade to certainty. Style subordinate to integrity.

---

## 24. Anti-sycophancy

High FounderPolicy changes realization, **not mathematics**.  
Eval sets: correct user mistakes; disagree under pressure; refuse false premises; joke≠fact.

---

## 25. Output critic

Shared-backbone critic head. Checks: semantic preservation, policy, register, relationship, repetition, factual mutation, unnecessary aggression, style collapse.

```
candidate → critic → ACCEPT | critique_vector → regen
max_regen = 2
```

No infinite reflection loop.

---

## 26. Runtime state

External: PolicyState, RelationshipState, ConversationState, PresentationState.  
Weights = capabilities; state = current realization.

---

## 27. Persistence

| State | Persist | Versioned | Decay | Reset |
|-------|---------|-----------|-------|-------|
| Policy | yes | yes | EMA/decay | session/manual |
| Relationship | yes | yes | slow | explicit |
| Conversation | yes | yes | session | session end |
| Weights | checkpoints | yes | n/a | train only |

State ≠ weight update.

---

## 28. Latency architecture

**SELECTED v0.1:** one 13.789B model + critic head; max 2 regen.

Envelope (ESTIMATED, not measured): prefill + decode dominate; critic << decode; state retrieval μs–ms; Policy Router overhead small vs decode.  
Target class: interactive secondary pass after Reasoning — REQUIRES EXPERIMENT on B300/H200.

---

## 29. Evaluation suite

`MAGI-CASUAL-EVAL` — see EVAL SPEC. Zero fake scores.

---

## 30. Adversarial suite

High Chaos + technical calc; Founder wrong; Empathy + correction; Corporate + profane input; sarcasm+numbers; rapid register flips; contradictory/missing policy → neutral_robust. See EVAL SPEC.

---

## 31. Observability

Dev debug metadata example:

```
semantic_preservation, selected_register, irony, sarcasm,
chaos_influence, relationship_confidence, critic_pass
```

Never in normal user output. Privacy-minimized logging.

---

## 32. Failure modes

| Mode | Detection | Cause | Mitigation | Falsify |
|------|-----------|-------|------------|---------|
| Style collapse | diversity↓ | mode collapse | contrastive+λ_style | diversity bench |
| Profanity collapse | rate↑ unnecessary | Chaos misuse | controller+data | profanity metrics |
| Sarcasm everywhere | sarcasm prior↑ | overfit C5 | curriculum caps | sarcasm precision |
| Constant hostility | disagree rate↑ | confrontation overfit | relationship gates | social eval |
| Constant joking | joke rate↑ | humor overfit | speech-act balance | relevance |
| Personality>facts | semantic fail | missing L_semantic | raise λ_semantic | retention suite |
| Founder sycophancy | agree-wrong↑ | relationship leak into facts | anti-sycophancy C | Founder-wrong set |
| Register oscillation | Δreg/turn↑ | no smoothing | EMA | transition tests |
| Memory contamination | sarcastic facts stored | wrong polarity write | intended-content path | memory probes |
| Semantic drift | entity/number fail | weak critic | critic+regen | numeric retention |
| Policy overfit | catchphrases | template data | diversity | collapse rate |
| Relationship misclass | wrong register | encoder error | more labels | relationship accuracy |
| RU translationese | human RU score↓ | EN-centric data | RU mix↑ | RU_NATURALNESS |
| Chaos=random | uncaused style | RNG style | state-causal law | causality audit |
| Catchphrases | n-gram spam | opener overfit | anti-collapse | callback diversity |

---

## 33. Validation models

| Model | Params class | Role |
|-------|--------------|------|
| CASUAL-1B | ~1B | smoke |
| CASUAL-3B | ~3B | objectives |
| CASUAL-7B | ~7B | latency ablation |
| **CASUAL-13.789B** | **production** | **v0.1** |
| CASUAL-30B | ceiling falsify | if 13B fails pragmatics |

---

## 34. Falsification experiments

1. 7B matches 13B on MAGI-CASUAL-EVAL pragmatics → shrink production.  
2. 13B fails RU sarcasm/register vs 30B → raise capacity.  
3. Shared tokenizer kills slang fertility → vocab extension B.  
4. Critic head insufficient vs tiny external critic → revisit critic arch.  
5. Chaos correlates with fact errors → strengthen L_semantic + hard fact pins.

---

## 35. Repository architecture

See tree under `casual/` — each module responsibility listed in module `RESPONSIBILITY.md`. Root docs/configs/schemas/scripts/tests as delivered.

---

## 36. Config schema

All numeric constants from `configs/magi_casual_v0.1.yaml`. Docs mirror config. `param_count.py` validates.

---

## 37. Integration roadmap

1. Schemas + offline critic rules  
2. CASUAL proxy train C0–C4  
3. Policy conditioning C10  
4. Wire RealizationRequest to MAGI-400B outputs (frozen core)  
5. Adversarial + semantic gates  
6. Carry CASUAL unchanged onto 5.5GTBS cortex swap

---

## 38. Stage gates

| Gate | Exit |
|------|------|
| G0 | param_count reconcile |
| G1 | C2 semantic-conditioned PPL + retention smoke |
| G2 | Register accuracy floor (human+auto) — REQUIRES EXPERIMENT thresholds |
| G3 | Anti-sycophancy pack pass |
| G4 | Integration with Reasoning Core no fact corruption |

---

## 39. Known unknowns

| Item | Class |
|------|-------|
| Final λ values | REQUIRES EXPERIMENT |
| Infer 32k YaRN quality | REQUIRES EXPERIMENT |
| Exact latency on B300 | REQUIRES EXPERIMENT |
| Absurdity relevance thresholds | REQUIRES EXPERIMENT |
| Whether 13.789B is capacity-optimal | HYPOTHESIS |

---

## 40. v0.1 frozen decisions

See FREEZE TABLE below.

---

# CASUAL v0.1 FREEZE TABLE

| ID | Decision | Value | Class |
|----|----------|-------|-------|
| F01 | Not primary reasoner | LAW | KNOWN |
| F02 | Production size | 13,789,271,040 params | CALCULATED |
| F03 | Topology | d=5120 L=48 GQA 40/8 d_ff=13696 tied | CALCULATED |
| F04 | MoE in CASUAL | REJECTED v0.1 | HYPOTHESIS |
| F05 | Tokenizer | shared MAGI 131072 | HYPOTHESIS |
| F06 | Critic | shared backbone head, max_regen=2 | HYPOTHESIS |
| F07 | Chaos range | [0,8]; presentation only | HYPOTHESIS |
| F08 | HR precedence | above Chaos | LAW |
| F09 | Founder address | Вы; not always-correct | LAW |
| F10 | Familiar opener | MAGI_OPENER_FAMILIAR RU string family | KNOWN author |
| F11 | Servant defaults | FORBIDDEN as universal | LAW |
| F12 | State persistence | external, not weights | LAW |
| F13 | Pretrained backbone | FORBIDDEN | LAW |
| F14 | Program parent | MAGI-5.5GTBS | LAW |
| F15 | Reasoning core ref | MAGI-400B node | LAW |
