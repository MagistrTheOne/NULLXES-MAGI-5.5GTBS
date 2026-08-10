#!/usr/bin/env python3
"""T4 smoke config + tokenizer validation tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from magi.config import load_model_config
from magi.model import MAGITransformer
from magi.tokenizer import MagiByteBPETokenizer, build_t4_smoke_tokenizer, train_byte_bpe

ROOT = Path(__file__).resolve().parents[1]


class TestT4Smoke(unittest.TestCase):
    def test_config_loads_and_bounds(self):
        cfg = load_model_config(ROOT / "configs" / "magi_t4_smoke_v0.1.yaml")
        self.assertEqual(cfg.name, "MAGI-T4-SMOKE")
        self.assertTrue(cfg.is_moe)
        self.assertEqual(cfg.vocab_size, 8192)
        self.assertEqual(cfg.n_heads * cfg.d_head, cfg.d_model)
        model = MAGITransformer(cfg)
        params = model.parameter_count()
        self.assertLess(params, 500_000_000)
        self.assertGreater(params, 50_000_000)
        # fp16 weight budget under 2GB for T4 headroom
        self.assertLess(params * 2, 2 * 1024**3)

    def test_tokenizer_roundtrip_and_vocab(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "tok.json"
            tok = train_byte_bpe(
                [
                    "MAGI T4 smoke tokenizer",
                    "проверка UTF-8 русский текст",
                    "def forward(x): return x",
                ],
                vocab_size=1024,
                tokenizer_id="unit",
            )
            tok.save(artifact)
            loaded = MagiByteBPETokenizer.load(artifact)
            text = "MAGI проверка forward"
            ids = loaded.encode(text, add_bos=True, add_eos=True)
            decoded = loaded.decode(ids)
            self.assertEqual(loaded.vocab_size, 1024)
            self.assertIn("MAGI", decoded)
            self.assertTrue(ids[0] == loaded.bos_id)
            self.assertTrue(ids[-1] == loaded.eos_id)

    def test_build_t4_smoke_tokenizer_matches_model_vocab(self):
        cfg = load_model_config(ROOT / "configs" / "magi_t4_smoke_v0.1.yaml")
        tok = build_t4_smoke_tokenizer(vocab_size=cfg.vocab_size)
        self.assertEqual(tok.vocab_size, cfg.vocab_size)
        ids = tok.encode("NULLXES MAGI", add_bos=True)
        self.assertTrue(all(0 <= i < cfg.vocab_size for i in ids))

    def test_cpu_forward_smoke(self):
        cfg = load_model_config(ROOT / "configs" / "magi_t4_smoke_v0.1.yaml")
        model = MAGITransformer(cfg)
        model.eval()
        ids = torch.randint(0, cfg.vocab_size, (1, 16), dtype=torch.long)
        with torch.no_grad():
            logits = model(ids)
        self.assertEqual(tuple(logits.shape), (1, 16, cfg.vocab_size))


if __name__ == "__main__":
    unittest.main()
