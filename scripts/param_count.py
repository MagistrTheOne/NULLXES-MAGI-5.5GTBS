#!/usr/bin/env python3
"""MAGI / MAGI parameter accounting — Python stdlib only.

Usage:
  python scripts/param_count.py --config configs/magi_400b_v0.1.yaml
  python scripts/param_count.py --config configs/magi_casual_v0.1.yaml
  python scripts/param_count.py --all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    out = []
    i = 0
    while i < len(line):
        ch = line[i]
        if ch == "'" and not in_double:
            in_single = not in_single
            out.append(ch)
        elif ch == '"' and not in_single:
            in_double = not in_double
            out.append(ch)
        elif ch == "#" and not in_single and not in_double:
            break
        else:
            out.append(ch)
        i += 1
    return "".join(out).rstrip()


def _parse_scalar(raw: str):
    s = raw.strip()
    if s == "" or s == "null" or s == "~":
        return None
    if s in ("true", "True"):
        return True
    if s in ("false", "False"):
        return False
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        parts = [p.strip() for p in inner.split(",")]
        return [_parse_scalar(p) for p in parts]
    try:
        if s.startswith("0x"):
            return int(s, 16)
        if "." in s or "e" in s.lower():
            return float(s)
        return int(s)
    except ValueError:
        return s


def load_simple_yaml(path: Path) -> dict:
    """Minimal indentation-based YAML subset loader (stdlib)."""
    text = path.read_text(encoding="utf-8")
    root: dict = {}
    stack: list[tuple[int, object]] = [(-1, root)]

    for lineno, raw_line in enumerate(text.splitlines(), 1):
        line = _strip_comment(raw_line)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if line.lstrip().startswith("- "):
            raise ValueError(f"{path}:{lineno}: list items not supported in this loader")
        if ":" not in line:
            raise ValueError(f"{path}:{lineno}: expected key:")
        key, _, rest = line.strip().partition(":")
        key = key.strip()
        rest = rest.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if not isinstance(parent, dict):
            raise ValueError(f"{path}:{lineno}: invalid parent for key {key}")

        if rest == "":
            node: dict = {}
            parent[key] = node
            stack.append((indent, node))
        else:
            parent[key] = _parse_scalar(rest)
    return root


def gqa_attn_params(d_model: int, n_kv_heads: int, d_head: int) -> int:
    d_kv = n_kv_heads * d_head
    w_q = d_model * d_model
    w_k = d_model * d_kv
    w_v = d_model * d_kv
    w_o = d_model * d_model
    return w_q + w_k + w_v + w_o


def swiglu_ffn_params(d_model: int, d_ff: int) -> int:
    return 3 * d_model * d_ff


def rmsnorm_params(d_model: int, n_layers: int, final: bool = True) -> int:
    # pre-attn + pre-ffn per layer (+ optional final)
    return n_layers * 2 * d_model + (d_model if final else 0)


def count_moe(cfg: dict) -> dict:
    arch = cfg["architecture"]
    d = int(arch["d_model"])
    L = int(arch["n_layers"])
    L_d = int(arch["n_dense_layers"])
    L_m = int(arch["n_moe_layers"])
    if L_d + L_m != L:
        raise ValueError(f"n_dense_layers + n_moe_layers != n_layers ({L_d}+{L_m}!={L})")
    n_kv = int(arch["n_kv_heads"])
    d_head = int(arch["d_head"])
    n_heads = int(arch["n_heads"])
    if n_heads * d_head != d:
        raise ValueError("n_heads * d_head must equal d_model")
    V = int(arch["vocab_size"])
    tied = bool(arch["tied_embeddings"])
    d_ff_d = int(arch["d_ff_dense"])
    d_ff_e = int(arch["d_ff_expert"])
    E_r = int(arch["n_routed_experts"])
    E_s = int(arch["n_shared_experts"])
    top_k = int(arch["top_k"])

    p_emb = V * d
    p_lm = 0 if tied else V * d
    p_attn_l = gqa_attn_params(d, n_kv, d_head)
    p_attn = L * p_attn_l
    p_dense_l = swiglu_ffn_params(d, d_ff_d)
    p_dense = L_d * p_dense_l
    p_expert = swiglu_ffn_params(d, d_ff_e)
    p_router_l = d * E_r
    p_moe_l = (E_r + E_s) * p_expert + p_router_l
    p_moe = L_m * p_moe_l
    p_norm = rmsnorm_params(d, L, final=True)
    total = p_emb + p_lm + p_attn + p_dense + p_moe + p_norm

    active_experts = top_k + E_s
    p_active_moe = L_m * (active_experts * p_expert + p_router_l)
    active = p_attn + p_dense + p_active_moe + p_lm + p_norm

    return {
        "model_class": "moe_decoder",
        "components": {
            "embeddings": p_emb,
            "lm_head": p_lm,
            "attention_all_layers": p_attn,
            "dense_ffn": p_dense,
            "moe_all_layers": p_moe,
            "norms": p_norm,
            "attn_per_layer": p_attn_l,
            "dense_ffn_per_layer": p_dense_l,
            "expert": p_expert,
            "moe_per_layer": p_moe_l,
            "router_per_layer": p_router_l,
        },
        "total": total,
        "active_per_token": active,
        "active_experts_per_moe_layer": active_experts,
        "cognitive_cycle_1pass": active,
        "cognitive_cycle_2pass": active * 2,
    }


def count_dense(cfg: dict) -> dict:
    arch = cfg["architecture"]
    d = int(arch["d_model"])
    L = int(arch["n_layers"])
    n_kv = int(arch["n_kv_heads"])
    d_head = int(arch["d_head"])
    n_heads = int(arch["n_heads"])
    if n_heads * d_head != d:
        raise ValueError("n_heads * d_head must equal d_model")
    V = int(arch["vocab_size"])
    tied = bool(arch["tied_embeddings"])
    d_ff = int(arch["d_ff"])

    p_emb = V * d
    p_lm = 0 if tied else V * d
    p_attn_l = gqa_attn_params(d, n_kv, d_head)
    p_attn = L * p_attn_l
    p_ffn_l = swiglu_ffn_params(d, d_ff)
    p_ffn = L * p_ffn_l
    p_norm = rmsnorm_params(d, L, final=True)
    total = p_emb + p_lm + p_attn + p_ffn + p_norm
    active = total  # dense: all params active (lm tied => emb counted once)

    return {
        "model_class": "dense_decoder",
        "components": {
            "embeddings": p_emb,
            "lm_head": p_lm,
            "attention_all_layers": p_attn,
            "ffn_all_layers": p_ffn,
            "norms": p_norm,
            "attn_per_layer": p_attn_l,
            "ffn_per_layer": p_ffn_l,
        },
        "total": total,
        "active_per_token": active,
        "cognitive_cycle_1pass": active,
        "cognitive_cycle_2pass": active * 2,
    }


def human(n: int) -> str:
    if n >= 1_000_000_000_000:
        return f"{n / 1_000_000_000_000:.3f}T"
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.3f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.3f}M"
    return str(n)


def analyze(path: Path) -> dict:
    cfg = load_simple_yaml(path)
    model_class = cfg.get("meta", {}).get("model_class")
    if model_class == "moe_decoder":
        result = count_moe(cfg)
    elif model_class == "dense_decoder":
        result = count_dense(cfg)
    else:
        raise ValueError(f"Unsupported meta.model_class: {model_class}")
    result["config"] = str(path)
    result["name"] = cfg.get("meta", {}).get("name")
    result["claim_class"] = "CALCULATED"
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="MAGI param accounting (stdlib)")
    parser.add_argument("--config", type=Path, help="Path to YAML config")
    parser.add_argument("--all", action="store_true", help="Count MAGI-35B, MAGI-400B, and MAGI-CASUAL")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args(argv)

    root = Path(__file__).resolve().parents[1]
    paths: list[Path] = []
    if args.all:
        paths = [
            root / "configs" / "magi_35b_moe_v0.1.yaml",
            root / "configs" / "magi_400b_v0.1.yaml",
            root / "configs" / "magi_casual_v0.1.yaml",
        ]
    elif args.config:
        paths = [args.config if args.config.is_absolute() else root / args.config]
    else:
        parser.error("Provide --config or --all")

    reports = [analyze(p) for p in paths]
    if args.json:
        print(json.dumps(reports, indent=2))
    else:
        for r in reports:
            print(f"=== {r['name']} ({r['model_class']}) ===")
            print(f"config: {r['config']}")
            for k, v in r["components"].items():
                print(f"  {k:24s} {v:18d}  ({human(v)})")
            print(f"  {'TOTAL':24s} {r['total']:18d}  ({human(r['total'])})")
            print(f"  {'ACTIVE/token':24s} {r['active_per_token']:18d}  ({human(r['active_per_token'])})")
            if "active_experts_per_moe_layer" in r:
                print(f"  active experts/MoE layer: {r['active_experts_per_moe_layer']}")
            print(f"  cognitive 1-pass: {human(r['cognitive_cycle_1pass'])}")
            print(f"  cognitive 2-pass: {human(r['cognitive_cycle_2pass'])}")
            print(f"  claim_class: {r['claim_class']}")
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
