#!/usr/bin/env python3
"""MAGI production casual chat REPL — MAGI weights only.

Identity: MAGI — Synthetic Intelligence System (NULLXES).
No OpenAI / Grok / Claude / Gemini / foreign from_pretrained backbone.
No smoke tokenizer fallback. No random-weight inference.

Usage:
  python scripts/magi_chat.py \\
    --config configs/magi_35b_moe_v0.1.yaml \\
    --checkpoint artifacts/magi_35b_moe/model.safetensors \\
    --tokenizer tokenizer/artifacts/magi_tokenizer_v0.1.json \\
    --device cuda

Smoke / T4 fixtures: scripts/dev/smoke_chat.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from magi.runtime.generate import GenerateConfig, generate
from magi.runtime.production import (
    build_chat_input_ids,
    context_limit,
    load_production_model_bundle,
)

IDENTITY = "MAGI — Synthetic Intelligence System (NULLXES)"


def main() -> int:
    parser = argparse.ArgumentParser(description="MAGI production native chat")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Required MAGI checkpoint (model.safetensors or step dir). Random init forbidden.",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="MAGI tokenizer artifact. If omitted, resolved from checkpoint meta / config artifact.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
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
    print(f"identity={IDENTITY}")
    print(f"model={cfg.name} moe={cfg.is_moe}")
    print("commands: /quit  /clear")
    print("---")

    ctx = context_limit(cfg)
    max_prompt = max(8, ctx - int(args.max_new_tokens))
    turns: list[tuple[str, str]] = []

    while True:
        try:
            user = input("User:\n").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user in {"/quit", "/exit", ":q"}:
            break
        if user == "/clear":
            turns.clear()
            print("history=cleared")
            continue

        turns.append(("user", user))
        # Branding prefix only — not a foreign-model system prompt.
        ids = build_chat_input_ids(
            tokenizer,
            identity=IDENTITY,
            turns=turns,
            max_prompt_tokens=max_prompt,
        )
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        result = generate(
            model,
            input_ids,
            config=GenerateConfig(
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                do_sample=True,
                eos_token_id=tokenizer.eos_id,
                pad_token_id=tokenizer.pad_id,
            ),
        )
        new_ids = result.new_token_ids[0].tolist()
        # Drop trailing EOS from the assistant turn for display/history.
        if new_ids and new_ids[-1] == tokenizer.eos_id:
            new_ids = new_ids[:-1]
        reply = tokenizer.decode(new_ids).strip()
        turns.append(("magi", reply))
        print(f"MAGI:\n{reply}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
