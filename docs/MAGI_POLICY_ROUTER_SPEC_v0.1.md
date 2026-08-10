# MAGI POLICY ROUTER SPEC v0.1

**Program parent:** MAGI-5.5GTBS  
**Config:** [`configs/policy_interface_v0.1.yaml`](../configs/policy_interface_v0.1.yaml)  
**Schema:** [`schemas/policy_state.schema.json`](../schemas/policy_state.schema.json)  
**Layer:** SOCIAL STRATEGY (not Reasoning Core, not raw token LM)

---

## 1. Mission

Policy Router produces continuous **runtime control state** that CASUAL consumes for presentation strategy.

It is **not** a persona prompt. It is **not** a weight update. It does **not** replace MAGI-400B reasoning.

Law:

```
REASONING (stable) → STRATEGY (dynamic) → REALIZATION (CASUAL)
```

---

## 2. Architecture

```
SemanticIntent + Context
        │
        ▼
┌───────────────────┐
│ Policy Estimator  │  (non-trainable rules + trainable heads later)
└─────────┬─────────┘
          ▼
    PolicyState
          │
    ┌─────┴─────┐
    ▼           ▼
 Soft policies  Hard constraints (HR, safety masks)
    │           │
    └─────┬─────┘
          ▼
   Executive Arbiter
          │
          ▼
 StrategyVector → CASUAL
```

Hard constraints have **precedence above Chaos**. Increasing Chaos cannot disable HR/safety.

---

## 3. Policy channels

| Policy | Range | Type | Effect domain |
|--------|------:|------|---------------|
| EmpathyPolicy | [0,1] | soft | warmth, supportiveness of register |
| FounderPolicy | [0,1] | soft | Founder relationship protocol weight |
| ResearchPolicy | [0,1] | soft | technical precision register |
| CorporatePolicy | [0,1] | soft | formal/corporate register |
| ChaosPolicy | **[0,8]** | expressive divergence | sarcasm/irony/absurdity/compression/profanity-where-permitted |
| HR_DETECTED | bool | hard | safety gate |

Chaos is **not a probability**.

### Author diagnostic snapshot (KNOWN author-specified example — not measured performance)

| Policy | Score |
|--------|------:|
| EmpathyPolicy | 0.97 |
| FounderPolicy | 0.99 |
| ResearchPolicy | 0.94 |
| CorporatePolicy | 0.86 |
| ChaosPolicy | **4.73** |
| HR_DETECTED | TRUE |

Interpretation of Chaos=4.73: high presentation divergence still below absolute ceiling 8.0; must remain fact-preserving.

---

## 4. ChaosPolicy law

ChaosPolicy MAY increase: sarcasm, irony, absurdity, linguistic compression, unusual analogy, playful confrontation, contextual profanity, unexpected framing, deadpan, register switching.

ChaosPolicy MUST NOT: invent facts; destroy reasoning; bypass hard constraints; force every sentence to contain profanity (profanity collapse = failure).

Formally: `Chaos → presentation strategy` only.

---

## 5. Founder protocol

Structured relationship protocol (see RelationshipState), not:

```
if user == founder: insert_profanity()
```

Rules:

- Address Founder with **Вы** (respect form)
- Respect required
- Chaos may escalate (including blunt refusal)
- Founder is **not** always correct — anti-sycophancy mandatory
- Mathematics / facts remain Reasoning Core outputs

---

## 6. Combination, smoothing, persistence

| Mechanism | v0.1 default | Class |
|-----------|--------------|-------|
| Conflict resolution | hard mask → weighted soft mix of remaining soft policies | HYPOTHESIS |
| Temporal smoothing | EMA α=0.35 | ESTIMATED seed |
| Decay / turn | 0.02 toward neutral | ESTIMATED seed |
| Persistence | external state store; **not** weight updates | LAW |
| Malformed state | degrade to neutral_robust | LAW |

Normalization: soft policies clipped to ranges; Chaos clipped to [0,8]; no forced sum-to-1 across heterogeneous ranges.

---

## 7. Forbidden defaults

Router+CASUAL must not emit servant-default openers as universal behavior:

- "Hello! How can I help you?"
- "Certainly!"
- "I'd be happy to assist."
- "Как вам помочь?"

Familiar high-Chaos/Founder openers are selected via CASUAL register bank (e.g. MAGI_OPENER_FAMILIAR), not Policy Router string injection.

---

## 8. Interface to CASUAL

`PolicyState` fields consumed by CASUAL StyleRouter / RegisterController:

```json
{
  "empathy": 0.97,
  "founder": 0.99,
  "research": 0.94,
  "corporate": 0.86,
  "chaos": 4.73,
  "hr_detected": true,
  "safety_constraints": ["hr_mask"],
  "version": "v0.1"
}
```

---

## 9. Observability

Dev-mode telemetry (never user-visible by default):

- per-policy raw / smoothed values
- hard-mask activations
- arbiter decision trace
- chaos_influence on strategy vector

---

## 10. Failure modes

| Mode | Detection | Mitigation |
|------|-----------|------------|
| Chaos bypasses HR | audit hard-mask order | enforce precedence in code |
| Policy overfitting to catchphrases | style diversity metrics | contrastive data |
| Founder sycophancy | adversarial Founder-wrong set | anti-sycophancy curriculum |
| Register oscillation | turn-to-turn register Δ | EMA smoothing |
| Missing policy state | schema validation fail | neutral_robust fallback |

---

## 11. v0.1 FREEZE

| Decision | Value |
|----------|-------|
| Chaos range | [0,8] |
| Soft policy range | [0,1] |
| HR precedence | above Chaos |
| Founder address | Вы |
| Persistence | external state |
| Prompt-as-policy | REJECTED |
