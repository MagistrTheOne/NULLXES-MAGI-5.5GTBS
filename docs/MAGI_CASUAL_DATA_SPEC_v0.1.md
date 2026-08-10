# MAGI CASUAL DATA SPEC v0.1

**Program parent:** MAGI-5.5GTBS  
**Schema:** [`schemas/casual_training_record.schema.json`](../schemas/casual_training_record.schema.json)  
**From-zero corpus:** NULLXES-approved only. No fabricated licenses.

---

## 1. Pipeline

```
RAW → provenance → legal/license filter → parse → normalize
  → language detection → quality score → exact dedup → fuzzy dedup
  → contamination filter → safety/privacy filter → domain classify
  → curriculum weight → tokenize (shared MAGI) → packed shards
```

Every shard carries lineage fields where practical. Unknown license ⇒ `license_known=false` and quarantine policy.

---

## 2. Taxonomy (minimum)

| Category | Intent |
|----------|--------|
| CASUAL_DIALOGUE | informal multi-turn |
| FORMAL_DIALOGUE | high formality |
| TECHNICAL_DIALOGUE | engineering/science talk |
| ARGUMENT / DEBATE | structured disagreement |
| SARCASM / IRONY | pragmatic polarity flip |
| ABSURDIST_HUMOR / DEADPAN | controlled incongruity |
| FRIENDLY_BANTER | warm teasing |
| CONFRONTATIONAL_DIALOGUE | challenge without fact break |
| PROFANITY_IN_CONTEXT | contextual, role-labeled |
| REGISTER_SWITCHING | within-session shifts |
| RELATIONSHIP_AWARE_DIALOGUE | Founder/colleague/unknown |
| LONG_RUNNING_CONVERSATION | continuity |
| CALLBACK_HUMOR | delayed references |
| SOCIAL_CORRECTION | correct without servant tone |
| DISAGREEMENT / REFUSAL / QUESTIONING | speech acts |
| CORPORATE / RESEARCH_DISCUSSION | policy-aligned registers |
| RU_INFORMAL / EN_INFORMAL / RU_EN_CODE_SWITCHING | multilingual |

Profanity is never an isolated word list; context + role required.

---

## 3. Record serialization

Training records MUST match `casual_training_record.schema.json`:

- `conversation_context`
- `semantic_intent`
- `reasoning_summary`
- `relationship_state`
- `policy_state`
- `target_strategy` / `target_register` / `target_response`
- `pragmatic_labels`
- optional `contrastive_group_id`

Internal machine fields are conditioning inputs — not user-visible chat text pretending to be architecture.

**REJECTED:** prepending `You are sarcastic.` as the training method.

---

## 4. Contrastive packs

For each `contrastive_group_id`:

1. Fix `semantic_intent` + `reasoning_summary` + facts/numbers.
2. Vary `policy_state` and/or `relationship_state`.
3. Provide multiple valid `target_response` realizations.
4. Judge invariant: numbers, negation, entities, uncertainty.

Example family:

| State | Realization class |
|-------|-------------------|
| Corporate | precise formal correction |
| Research | accounting/recompute language |
| Founder+Chaos high | blunt familiar RU |

---

## 5. Semantic pins

Each record may attach explicit pins used by L_semantic:

- facts[]
- numbers[{name,value}]
- negation flags
- uncertainty level
- required action

CASUAL targets must preserve pins.

---

## 6. Mixture seeds (ESTIMATED)

| Bucket | Weight seed |
|--------|------------:|
| general casual dialogue | 0.25 |
| technical/research | 0.15 |
| contrastive register packs | 0.20 |
| sarcasm/irony/absurd | 0.10 |
| disagreement/anti-sycophancy | 0.10 |
| RU informal + code-switch | 0.10 |
| long-context continuity | 0.05 |
| safety/HR hard constraint demos | 0.05 |

Final mixture = REQUIRES EXPERIMENT.

---

## 7. Quality / safety filters

- PII scrub where required
- Contamination vs eval held-out
- HR/hard-constraint demos labeled, not removable by Chaos
- No invented benchmark leakage

---

## 8. Provenance policy

| Status | Action |
|--------|--------|
| license known + approved | admit |
| license unknown | flag + quarantine |
| prohibited | reject |
| synthetic NULLXES-generated | mark `source_id=nullxes_synth` |

Do not fabricate provenance.

---

## 9. v0.1 FREEZE

| Decision | Value |
|----------|-------|
| Schema | casual_training_record v0.1 |
| Contrastive primitive | mandatory |
| Prompt-persona data as architecture | REJECTED |
| Shared tokenizer | YES |
