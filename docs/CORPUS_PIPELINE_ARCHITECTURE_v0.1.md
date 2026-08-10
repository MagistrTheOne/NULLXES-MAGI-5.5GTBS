# CORPUS PIPELINE ARCHITECTURE v0.1

**Program:** MAGI-5.5GTBS  
**Config:** [`configs/corpus_pipeline_v0.1.yaml`](../configs/corpus_pipeline_v0.1.yaml)  
**Schemas:** [`schemas/dataset_manifest.schema.json`](../schemas/dataset_manifest.schema.json), [`schemas/training_shard_manifest.schema.json`](../schemas/training_shard_manifest.schema.json)  
**Dataset content:** provided later by author.

---

## 1. Pipeline

```mermaid
flowchart LR
  Raw[RawIngest] --> Prov[Provenance]
  Prov --> Legal[LegalFilter]
  Legal --> Parse[Parser]
  Parse --> Norm[Normalizer]
  Norm --> Lang[LangID]
  Lang --> Quality[QualityScore]
  Quality --> Exact[ExactDedup]
  Exact --> Fuzzy[FuzzyDedup]
  Fuzzy --> Contam[ContaminationFilter]
  Contam --> Privacy[PrivacyFilter]
  Privacy --> Domain[DomainClassifier]
  Domain --> Mix[CurriculumWeight]
  Mix --> Tok[Tokenizer]
  Tok --> Pack[SequencePacking]
  Pack --> Shard[ShardWriter]
  Shard --> Manifest[ManifestWriter]
```

---

## 2. Non-Negotiables

- Unknown license goes to quarantine.
- Provenance is mandatory.
- Dataset licenses are never fabricated.
- Synthetic data requires generator manifest and semantic pins.
- External LLM runtime dependency is forbidden for MAGI cognition.
- Benchmark contamination filtering is required before eval release.

---

## 3. Synthetic Data Interface

Synthetic datasets enter as `source_type=nullxes_synthetic` and must include:

| Field | Purpose |
|-------|---------|
| `generator_id` | reproducibility |
| `semantic_pins` | fact/number/logic preservation |
| `prompt_family` | curriculum control |
| `license_status=NULLXES_SYNTHETIC` | ownership |
| `contamination_report` | eval hygiene |

The actual synthetic dataset taxonomy is intentionally deferred to the author.

---

## 4. Shard Contract

Training shards are packed-token binaries plus JSON manifest:

- tokenizer hash;
- config hash;
- sequence length;
- token count;
- document count;
- dataset lineage weights;
- quality summary;
- shard hash.

Document boundaries are preserved where possible; EOS is inserted between documents.

---

## 5. Curriculum Domains

General language, science, mathematics, engineering, programming, systems, robotics, economics, law, history, philosophy, dialogue, reasoning, structured data, multilingual, casual dialogue, policy-conditioned dialogue.

Weights are config-driven. No final weights until datasets exist.

---

## 6. Gates

| Gate | Blocks if |
|------|-----------|
| provenance | missing owner/capture metadata |
| license | unknown/rejected unless quarantine |
| dedup | high exact/fuzzy duplicate rate |
| contamination | overlap with eval |
| tokenizer | tokenizer hash mismatch |
| shard | manifest/hash mismatch |

---

## 7. v0.1 FREEZE

| Decision | Value |
|----------|-------|
| Dataset content | deferred to author |
| Pipeline shape | RAW→manifested packed shards |
| Synthetic accepted | yes, with manifest |
| License unknown | quarantine |
| Shard manifest | mandatory |
