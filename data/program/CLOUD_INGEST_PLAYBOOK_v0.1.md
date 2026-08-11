# Cloud-only ingest playbook — DO NOT run bulk download on local workstation.

meta:
  name: MAGI_DATA_CLOUD_INGEST
  version: v0.1
  local_workstation: FORBIDDEN_FOR_BULK
  cloud: RunPod_H200_or_B300_after_PILOT_approval
  approval_scope: PILOT_V0.1

prerequisites:
  - "Author signed approval_checklist for PILOT_V0.1"
  - "Author set MAGI_DATA_PILOT_APPROVED=1 on cloud (legacy MAGI_DATA_MANIFEST_APPROVED=1 = pilot only)"
  - "MAGI_DATA_PRODUCTION_APPROVED must NOT be set — production ingest does not exist yet"
  - "Reviewed data/program/MAGI_DATA_MANIFEST_v0.1.yaml"
  - "Reviewed data/program/MAGI_DATA_QUALITY_LAW_v0.1.md"
  - "HF token available on cloud only"
  - "Target disk sized for PILOT slices (~20–35 GB accepted text), NOT full vanity dumps"

pilot_targets_accepted_text_gb:
  fineweb2_ru: "5-10"
  fineweb2_en: "5-10"
  stack_python: "3-5"
  stack_typescript: "2-3"
  finemath: "2-3"
  wikipedia_ru_en: "2-3"
  nullxes_approved_slice: "available_approved_only"
  total_expected: "20-35"

phases:
  0_manifest_lock:
    action: "copy approved manifest to cloud workspace; freeze PILOT_V0.1 candidate list"
  1_license_matrix:
    action: "per-candidate license table; Stack v2 per-repo gate for python+typescript only"
  2_pilot_shards:
    action: "download PILOT slices only (GB-scale), never FineWeb2/Stack full dump"
    examples:
      - "FineWeb2: RU + EN sample configs only"
      - "Stack v2: python + typescript with per-repo license gate"
      - "FineMath: small slice"
      - "Wikipedia RU/EN: small factual slice"
  3_pipeline:
    action: "source → provenance → filter → exact/fuzzy/cross-source dedup → mixture → token stats → sharding"
  4_eval_gates:
    action: "raw/accepted docs, dedup rate, lang purity, quality hist, fertility, tokens/domain, contamination"
  5_tokenizer_experiments:
    action: "AFTER pilot ingest — vocab sweep 64k/96k/128k-131k/160k on fixed representative sample"
  6_production:
    action: "FORBIDDEN until separate MAGI_DATA_PRODUCTION_APPROVED exists and is signed"

forbidden:
  - "interpreting APPROVED=1 as download FineWeb2 entirely"
  - "Character / SFT / distillation into BASE"
  - "huggingface-cli download multi-TB on local workstation"
  - "production mix weights as approved numbers before pilot metrics"

cloud_command_templates:
  note: "Templates only — PILOT subsets after MAGI_DATA_PILOT_APPROVED=1"
  fineweb2_ru_pilot: >
    hf download HuggingFaceFW/fineweb-2 --repo-type dataset
    --include "data/rus_Cyrl/*" --local-dir /data/magi/raw/fineweb2_ru_pilot
  fineweb2_en_pilot: >
    hf download HuggingFaceFW/fineweb-2 --repo-type dataset
    --include "data/eng_Latn/*" --local-dir /data/magi/raw/fineweb2_en_pilot
  stack_v2_python_pilot: >
    hf download bigcode/the-stack-v2 --repo-type dataset
    --include "data/Python/*" --local-dir /data/magi/raw/stack_v2_py_pilot
  stack_v2_ts_pilot: >
    hf download bigcode/the-stack-v2 --repo-type dataset
    --include "data/TypeScript/*" --local-dir /data/magi/raw/stack_v2_ts_pilot
  gate_env: "export MAGI_DATA_PILOT_APPROVED=1"

outputs_expected:
  - /data/magi/raw/<candidate>/PROVENANCE.json
  - /data/magi/filtered/<bucket>/*.parquet
  - /data/magi/shards/train-*.bin + shards_manifest.json
  - reports/ingest_<candidate>_pilot.md
  - reports/pilot_metrics_v0.1.json
