# MAGI DATA PROGRAM v0.1

**Law:** [`MAGI_DATA_QUALITY_LAW_v0.1.md`](MAGI_DATA_QUALITY_LAW_v0.1.md)  
**Manifest:** [`MAGI_DATA_MANIFEST_v0.1.yaml`](MAGI_DATA_MANIFEST_v0.1.yaml)  
**Cloud ingest:** [`CLOUD_INGEST_PLAYBOOK_v0.1.md`](CLOUD_INGEST_PLAYBOOK_v0.1.md)

```text
BASE → CASUAL → CHARACTER
```

## Validate (local, no download)

```bash
python scripts/validate_data_manifest.py
```

## Layers

| Layer | Path | Notes |
|-------|------|-------|
| BASE | manifest `base_mixture_v0_1` | FineWeb2 / Dolma / Stack v2 / FineMath candidates |
| CASUAL | `casual/MAGI_CASUAL_MIX_v0.1.yaml` | sources TBD — privacy gated |
| CHARACTER | `character/` | human-curated; mass gen FORBIDDEN |

## Download policy

- Local workstation: **FORBIDDEN** for bulk corpora
- Cloud (H200/B300): only after `MAGI_DATA_MANIFEST_APPROVED=1`
- Row count / TB downloaded ≠ quality

## Author approval

Edit `approval_checklist` in the manifest, then on cloud:

```bash
export MAGI_DATA_MANIFEST_APPROVED=1
python scripts/data_ingest_gated.py --candidate fineweb2
```
