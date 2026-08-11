# MAGI DATA QUALITY LAW v0.1

**Program:** MAGI-5.5GTBS / NULLXES  
**Status:** NON-NEGOTIABLE  
**Download policy:** cloud ingest only after manifest approval. Local workstation MUST NOT download multi-GB corpora.

---

## Layers (order is law)

```text
MAGI BASE        → language / world / code / STEM
MAGI CASUAL      → how people actually talk
MAGI CHARACTER   → NULLXES behavioral identity (mode switching)
```

Character is **not** a bulk dialogue dump.  
Character is **behavioral dimensions + mode switching + preference pairs**.

---

## FORBIDDEN

```text
- bulk generation of synthetic dialogue for volume
- "generate N examples" pipelines to inflate rows
- recursive self-generation / model-eating-own-output as majority source
- synthetic paraphrase multiplication
- template mutation used to inflate row count
- one external LLM generating majority of MAGI personality data
- GPT / Grok / Claude / Gemini outputs treated as ground truth for CHARACTER
- duplicated semantic examples with cosmetic wording changes
- automatic personality injection into every sample
- catchphrase spam / profanity augmentation for volume
- artificial "reasoning traces" generated solely to increase token count
- downloading terabytes before PILOT approval
- treating APPROVED=1 as permission to pull FineWeb2 / Stack in full
- Character / SFT / distillation in BASE
- treating row count or GB downloaded as quality metrics
- T4/smoke fixture corpora as production training mixture
- approving production mix weights before pilot metrics
```

## REQUIRED

```text
- provenance for every shard
- license gate (unknown → quarantine)
- language ID + domain tags
- exact + fuzzy + cross-source dedup
- PII / secret / contamination filters
- mixture weights as EXPERIMENTAL_HYPOTHESIS until pilot metrics
- PILOT_V0.1 gate: MAGI_DATA_PILOT_APPROVED=1 (cloud only)
- production ingest requires a separate future approval (does not exist yet)
- CHARACTER: human-curated seed + pattern expansion; mass-gen BAN; not in BASE
- NEGATIVE character pairs (when sarcasm must turn OFF)
- tokenizer experiments only after pilot ingest on fixed sample
```

## Character rule

```text
If 10,000 high-quality human-curated character examples
outperform 5,000,000 synthetic examples:

USE 10,000.

ROW COUNT IS NOT A QUALITY METRIC.
```

## Intelligence before character

```text
casual banter     → sarcasm allowed
engineering bug   → dry irony allowed
distress/legal/finance/serious personal → sarcasm SUPPRESSED
uncertainty       → admit uncertainty
wrong premise     → disagree
MAGI error        → acknowledge + fix
```

First: model can predict language and reason.  
Then: register control.  
Then: MAGI character preferences.
