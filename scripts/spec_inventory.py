#!/usr/bin/env python3
"""Validate presence of MAGI-5.5GTBS architecture artifacts.

Python stdlib only. This does not validate JSON Schema semantics; it verifies
that the architecture pack expected by v0.2 exists.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REQUIRED_FILES = [
    "README.md",
    "configs/magi_35b_moe_v0.1.yaml",
    "configs/magi_7b_moe_v0.1.yaml",
    "configs/magi_7b_v0.1.yaml",
    "configs/magi_7b_train_h200_v0.1.yaml",
    "configs/tokenizer_bringup_8k_v0.1.yaml",
    "scripts/h200_train.py",
    "scripts/train_magi.py",
    "notebooks/RUNPOD_H200_INIT.md",
    "configs/magi_5_5gtbs_architecture_envelope_v0.2.yaml",
    "configs/cluster_profiles_v0.1.yaml",
    "configs/tokenizer_experiments_v0.1.yaml",
    "configs/corpus_pipeline_v0.1.yaml",
    "configs/training_profiles_v0.1.yaml",
    "configs/synthetic_magi_v0.1.yaml",
    "schemas/dataset_manifest.schema.json",
    "magi/data/__init__.py",
    "magi/data/synthetic/__init__.py",
    "magi/data/synthetic/build.py",
    "magi/data/synthetic/generators.py",
    "scripts/build_synthetic_dataset.py",
    "tests/test_synthetic_dataset.py",
    "data/synthetic/magi_synth_v0.1/golden/records.jsonl",
    "configs/casual_serving_profiles_v0.1.yaml",
    "configs/serving_profiles_v0.1.yaml",
    "configs/gate_thresholds_v0.1.yaml",
    "docs/MAGI-35B_MOE_ARCHITECTURE_SPEC_v0.1.md",
    "docs/MAGI_5.5GTBS_MASTER_ARCHITECTURE_SPEC_v0.2.md",
    "docs/CLUSTER_LAUNCH_ARCHITECTURE_v0.1.md",
    "docs/TOKENIZER_ARCHITECTURE_SPEC_v0.1.md",
    "docs/CORPUS_PIPELINE_ARCHITECTURE_v0.1.md",
    "docs/TRAINING_SYSTEM_ARCHITECTURE_v0.1.md",
    "docs/CASUAL_PRODUCTION_INTEGRATION_v0.1.md",
    "docs/INFERENCE_SERVING_ARCHITECTURE_v0.1.md",
    "docs/OBSERVABILITY_AND_GATES_v0.1.md",
    "docs/HF_COMPATIBILITY.md",
    "docs/CHECKPOINT_FORMAT.md",
    "docs/CONFIG_FORMAT.md",
    "schemas/training_shard_manifest.schema.json",
    "tokenizer/RESPONSIBILITY.md",
    "tokenizer/data/bringup_seed.txt",
    "tokenizer/artifacts/magi_bringup_8k_v0.1.json",
    "magi/tokenizer/byte_bpe.py",
    "scripts/validate_all_models.py",
    "magi/train/__init__.py",
    "magi/train/loop.py",
    "magi/train/single_gpu.py",
    "tests/test_all_model_configs_hf.py",
    "magi/__init__.py",
    "magi/config/loader.py",
    "magi/model/layers.py",
    "magi/model/moe.py",
    "magi/model/transformer.py",
    "magi/model/outputs.py",
    "magi/runtime/dry_init.py",
    "pyproject.toml",
    "magi/checkpoint/manifest.py",
    "magi/hf/__init__.py",
    "magi/hf/configuration_magi.py",
    "magi/hf/modeling_magi.py",
    "magi/hf/auto.py",
    "magi/hf/convert.py",
    "magi/hf/serialization.py",
    "magi/hf/generation.py",
    "magi/hf/versions.py",
    "scripts/validate_config.py",
    "scripts/dry_init.py",
    "scripts/cluster_preflight.py",
    "tests/test_magi_model_core.py",
    "tests/test_hf_compatibility.py",
    "tests/test_bringup_tokenizer.py",
    "tests/test_checkpoint_contract.py",
    "configs/tokenizer_bringup_8k_v0.1.yaml",
    "tokenizer/artifacts/magi_bringup_8k_v0.1.json",
]


def inventory(root: Path) -> tuple[list[Path], list[Path]]:
    found: list[Path] = []
    missing: list[Path] = []
    for rel in REQUIRED_FILES:
        path = root / rel
        if path.exists():
            found.append(path)
        else:
            missing.append(path)
    return found, missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MAGI spec inventory")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)

    found, missing = inventory(args.root)
    print(f"required={len(REQUIRED_FILES)} found={len(found)} missing={len(missing)}")
    for path in missing:
        print(f"MISSING {path.relative_to(args.root)}")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
