#!/usr/bin/env python3
"""MAGI production one-shot generate — MAGI weights only.

No smoke tokenizer fallback. Checkpoint required.

Usage:
  python scripts/magi_generate.py \\
    --config configs/magi_35b_moe_v0.1.yaml \\
    --checkpoint artifacts/magi_35b_moe/model.safetensors \\
    --tokenizer tokenizer/artifacts/magi_tokenizer_v0.1.json \\
    --prompt "The future of intelligence" \\
    --device cuda

Smoke fixtures: scripts/dev/smoke_generate.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from magi.runtime.generate import GenerateConfig, generate
from magi.runtime.production import load_production_model_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description="MAGI production native generate")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=None)
    parser.add_argument("--prompt", type=str, default="The future of intelligence")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    import torch

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    cfg, model, tokenizer, weights, tok_path = load_production_model_bundle(
        config_path=args.config,
        checkpoint=args.checkpoint,
        tokenizer_arg=args.tokenizer,
        device=device,
        root=ROOT,
    )
    print(f"loaded_weights={weights}")
    print(f"loaded_tokenizer={tok_path} id={tokenizer.tokenizer_id}")
    print(f"model={cfg.name}")

    ids = tokenizer.encode(args.prompt, add_bos=True)
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    result = generate(
        model,
        input_ids,
        config=GenerateConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=args.do_sample,
            eos_token_id=tokenizer.eos_id,
            pad_token_id=tokenizer.pad_id,
        ),
    )
    new_ids = result.new_token_ids[0].tolist()
    if new_ids and new_ids[-1] == tokenizer.eos_id:
        new_ids = new_ids[:-1]
    text = tokenizer.decode(new_ids)
    print("=== MAGI ===")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
