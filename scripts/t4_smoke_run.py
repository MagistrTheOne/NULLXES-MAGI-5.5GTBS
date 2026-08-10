#!/usr/bin/env python3
"""Validate MAGI T4 smoke config + tokenizer, then run CPU/CUDA forward.

Usage:
  python scripts/t4_smoke_run.py
  python scripts/t4_smoke_run.py --device cuda --seq 256
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from magi.config import load_model_config
from magi.tokenizer import build_t4_smoke_tokenizer


def main() -> int:
    parser = argparse.ArgumentParser(description="MAGI T4 smoke validation/run")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "magi_t4_smoke_v0.1.yaml")
    parser.add_argument("--device", default="cpu", help="cpu | cuda | cuda:0")
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--generate-tokens", type=int, default=8)
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args()

    cfg = load_model_config(args.config)
    print("=== CONFIG ===")
    print(f"name={cfg.name}")
    print(f"model_class={cfg.model_class}")
    print(f"d_model={cfg.d_model} n_layers={cfg.n_layers} vocab={cfg.vocab_size}")
    print(f"moe={cfg.is_moe} tied={cfg.tied_embeddings}")
    print(f"train_context={cfg.train_context} infer_context={cfg.infer_context}")

    tokenizer = build_t4_smoke_tokenizer(vocab_size=cfg.vocab_size)
    sample = "MAGI T4 smoke: проверка токенайзера и конфига."
    ids = tokenizer.encode(sample, add_bos=True, add_eos=True)
    decoded = tokenizer.decode(ids)
    roundtrip_ok = decoded == sample
    print("=== TOKENIZER ===")
    print(f"id={tokenizer.tokenizer_id}")
    print(f"vocab_size={tokenizer.vocab_size}")
    print(f"ids_len={len(ids)}")
    print(f"roundtrip_ok={roundtrip_ok}")
    print(f"decoded={decoded!r}")
    if tokenizer.vocab_size != cfg.vocab_size:
        raise SystemExit(f"tokenizer vocab {tokenizer.vocab_size} != model vocab {cfg.vocab_size}")
    if not roundtrip_ok:
        raise SystemExit("tokenizer roundtrip failed")

    # stdlib param count
    sys.path.insert(0, str(ROOT / "scripts"))
    import param_count

    report = param_count.analyze(Path(args.config))
    print("=== PARAMS ===")
    print(f"total={report['total']} ({param_count.human(report['total'])})")
    print(f"active={report['active_per_token']} ({param_count.human(report['active_per_token'])})")
    bytes_fp16 = report["total"] * 2
    print(f"fp16_weights_gb={bytes_fp16 / (1024**3):.3f}")
    if bytes_fp16 > 12 * (1024**3):
        raise SystemExit("T4 smoke weights exceed 12GB fp16 budget")

    if args.skip_model:
        print("model_run=skipped")
        return 0

    import torch
    from magi.hf import HF_AVAILABLE, HF_IMPORT_ERROR, MagiForCausalLM, native_config_to_hf
    from magi.model import MAGITransformer

    if not HF_AVAILABLE or MagiForCausalLM is None:
        raise SystemExit(f"magi.hf unavailable: {HF_IMPORT_ERROR!r}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("cuda requested but not available")

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print("=== MODEL ===")
    print(f"device={device} dtype={dtype}")

    hf_cfg = native_config_to_hf(cfg)
    hf_cfg.pad_token_id = tokenizer.pad_id
    hf_cfg.bos_token_id = tokenizer.bos_id
    hf_cfg.eos_token_id = tokenizer.eos_id
    model = MagiForCausalLM(hf_cfg)
    model.to(device=device, dtype=dtype)
    model.eval()

    native = MAGITransformer.from_config(cfg)
    print(f"native_params={native.parameter_count()}")
    print(f"hf_params={sum(p.numel() for p in model.parameters())}")

    prompt_ids = tokenizer.encode("MAGI smoke forward", add_bos=True)
    seq = min(args.seq, cfg.infer_context or args.seq)
    if len(prompt_ids) < seq:
        prompt_ids = prompt_ids + [tokenizer.pad_id] * (seq - len(prompt_ids))
    else:
        prompt_ids = prompt_ids[:seq]
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = (input_ids != tokenizer.pad_id).long()

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        print(f"logits_shape={tuple(out.logits.shape)}")
        print(f"cache_len={out.past_key_values.get_seq_length()}")
        gen = model.generate(
            input_ids=input_ids[:, : min(32, seq)],
            attention_mask=attention_mask[:, : min(32, seq)],
            max_new_tokens=args.generate_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_id,
            eos_token_id=tokenizer.eos_id,
        )
        text = tokenizer.decode(gen[0].tolist())
        print(f"generate_shape={tuple(gen.shape)}")
        print(f"generate_text={text!r}")

    if device.type == "cuda":
        allocated = torch.cuda.memory_allocated(device) / (1024**3)
        reserved = torch.cuda.memory_reserved(device) / (1024**3)
        print(f"cuda_allocated_gb={allocated:.3f}")
        print(f"cuda_reserved_gb={reserved:.3f}")
    print("status=OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
