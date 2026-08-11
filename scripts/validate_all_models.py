#!/usr/bin/env python3
"""Validate production MAGI model configs (native + HF meta path).

T4/smoke fixtures are NOT in the production set.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from magi.config import load_model_config
from magi.hf import HF_AVAILABLE, MagiForCausalLM, format_hf_import_error, native_config_to_hf
from magi.model import MAGITransformer
import param_count
import torch

if not HF_AVAILABLE or MagiForCausalLM is None:
    raise SystemExit(f"magi.hf unavailable: {format_hf_import_error()}")


PRODUCTION_CONFIGS = [
    ROOT / "configs" / "magi_7b_moe_v0.1.yaml",
    ROOT / "configs" / "magi_7b_v0.1.yaml",
    ROOT / "configs" / "magi_casual_v0.1.yaml",
    ROOT / "configs" / "magi_35b_moe_v0.1.yaml",
    ROOT / "configs" / "magi_400b_v0.1.yaml",
]


def validate_one(path: Path) -> None:
    cfg = load_model_config(path)
    report = param_count.analyze(path)
    hf_cfg = native_config_to_hf(cfg)

    required_aliases = {
        "num_hidden_layers": cfg.n_layers,
        "hidden_size": cfg.d_model,
        "num_attention_heads": cfg.n_heads,
        "num_key_value_heads": cfg.n_kv_heads,
        "head_dim": cfg.d_head,
    }
    for name, expected in required_aliases.items():
        got = getattr(hf_cfg, name)
        if int(got) != int(expected):
            raise RuntimeError(f"{path.name}: {name}={got} != {expected}")

    with torch.device("meta"):
        native = MAGITransformer.from_config(cfg)
        model = MagiForCausalLM(hf_cfg)
        model.tie_weights(recompute_mapping=False)

    native_n = native.parameter_count()
    hf_n = sum(p.numel() for p in model.parameters())
    if native_n != hf_n or native_n != report["total"]:
        raise RuntimeError(
            f"{path.name}: param mismatch native={native_n} hf={hf_n} report={report['total']}"
        )

    print(
        f"OK {cfg.name:18s} total={param_count.human(report['total']):>10s} "
        f"active={param_count.human(report['active_per_token']):>10s} "
        f"hf_aliases=ok tie_weights=ok"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate production MAGI model configs")
    parser.parse_args()

    for path in PRODUCTION_CONFIGS:
        if not path.exists():
            raise FileNotFoundError(path)
        validate_one(path)
    print("status=OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
