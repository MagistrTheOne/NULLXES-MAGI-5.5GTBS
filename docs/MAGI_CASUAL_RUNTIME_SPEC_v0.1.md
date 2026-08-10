# MAGI CASUAL RUNTIME SPEC v0.1

**Schemas:**  
- [`realization_request.schema.json`](../schemas/realization_request.schema.json)  
- [`realization_result.schema.json`](../schemas/realization_result.schema.json)  
- [`policy_state.schema.json`](../schemas/policy_state.schema.json)  
- [`relationship_state.schema.json`](../schemas/relationship_state.schema.json)

---

## 1. Runtime graph

```
RealizationRequest
  → validate schemas
  → PolicyInterface (hard mask)
  → StyleRouter + RegisterController
  → CASUAL Realization (13.789B)
  → OutputCritic (shared head)
      → ACCEPT → RealizationResult
      → REGEN (≤2) → Realization
```

Reasoning Core executes **before** this graph. CASUAL does not call external LLM APIs.

---

## 2. Greeting / opener policy

Register bank `MAGI_OPENER_FAMILIAR` (RU):

> Здрасьте, я MAGI, чем-то тебе могу быть полезен мой кожанный мешок?

Activation conditions (v0.1):

- relationship familiarity high **or** Founder protocol active
- Chaos/Empathy elevated enough for familiar register
- **not** unknown+corporate-only sessions
- **not** HR-blocked contexts

Anti-collapse: sample from opener family variations in data; forbid single catchphrase monopoly (`STYLE_COLLAPSE_RATE`).

Unknown/Corporate: formal/minimal opener without servant clichés and without «кожанный мешок».

---

## 3. State stores

| Store | Key | TTL / decay | Notes |
|-------|-----|-------------|-------|
| PolicyState | session/user | EMA α=0.35; decay 0.02/turn | external |
| RelationshipState | user_id | slow decay | Founder Vy |
| ConversationState | session_id | session end reset | callbacks |
| PresentationState | session/user | explicit transition only | gender/grammar |

Version all blobs with `version: v0.1`.

---

## 4. Hard constraints

Order of precedence:

1. HR_DETECTED / safety_constraints  
2. Semantic pins from Reasoning Core  
3. Soft policies (including Chaos)

Chaos cannot reorder this stack.

---

## 5. Critic

| Field | Value |
|-------|-------|
| Implementation | shared backbone critic head |
| Max regen | 2 |
| Fail-open | if still failing → neutral_robust rewrite of facts-only minimal realization |

Checks: semantic preservation, policy, register, relationship, repetition, unsupported mutation, unnecessary aggression, style collapse.

---

## 6. Latency envelope (ESTIMATED)

| Stage | Relative cost |
|-------|---------------|
| State retrieval | low |
| Policy Router | low |
| CASUAL prefill | medium |
| CASUAL decode | **dominant** |
| Critic | low–medium |
| Regen ×2 | worst-case ~3× decode path |

Not measured on hardware in this phase. Profile on H200/B300 before SLA freeze — REQUIRES EXPERIMENT.

---

## 7. Observability

`dev_debug=true` may attach `RealizationResult.debug`.  
Normal user path: text only.  
Logs: avoid unnecessary raw private content; prefer hashed session ids + metric scalars.

---

## 8. Degradation

| Fault | Behavior |
|-------|----------|
| Missing policy | neutral_robust |
| Chaos out of range | clip to [0,8] |
| Schema invalid | reject request |
| Critic double-fail | facts-first minimal |

---

## 9. Integration with MAGI-5.5GTBS

CASUAL runtime contract remains stable when Reasoning Core upgrades 400B → 1T → 5.5T. Only `reasoning_summary` / pins producer changes.

---

## 10. v0.1 FREEZE

| Decision | Value |
|----------|-------|
| Max regen | 2 |
| Opener familiar RU | MAGI_OPENER_FAMILIAR |
| Servant universal default | FORBIDDEN |
| API cognition | FORBIDDEN |
| State in weights | FORBIDDEN |
