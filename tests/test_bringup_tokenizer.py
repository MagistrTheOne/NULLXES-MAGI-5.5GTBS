"""Bring-up tokenizer unit tests (no T4/smoke)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from magi.config import load_model_config
from magi.tokenizer import (
    MagiByteBPETokenizer,
    build_bringup_tokenizer,
    load_bringup_tokenizer,
    train_byte_bpe,
)


class BringupTokenizerTests(unittest.TestCase):
    def test_bringup_artifact_matches_7b_moe_vocab(self):
        cfg = load_model_config(ROOT / "configs" / "magi_7b_moe_v0.1.yaml")
        tok = load_bringup_tokenizer(root=ROOT)
        self.assertEqual(tok.tokenizer_id, "magi_bringup_8k_v0.1")
        self.assertEqual(tok.vocab_size, cfg.vocab_size)
        self.assertEqual(tok.vocab_size, 8192)

    def test_roundtrip_ascii(self):
        tok = load_bringup_tokenizer(root=ROOT)
        text = "MAGI bring-up tokenizer roundtrip"
        ids = tok.encode(text, add_bos=True, add_eos=True)
        self.assertEqual(ids[0], tok.bos_id)
        self.assertEqual(ids[-1], tok.eos_id)
        decoded = tok.decode(ids)
        self.assertEqual(decoded, text)

    def test_train_refuses_smoke_id(self):
        with self.assertRaises(ValueError):
            train_byte_bpe(["hello"], vocab_size=300, tokenizer_id="magi_t4_smoke_v0.1")

    def test_build_bringup_from_seed_when_missing_artifact(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "tok.json"
            seed = ROOT / "tokenizer" / "data" / "bringup_seed.txt"
            tok = build_bringup_tokenizer(
                seed_path=seed,
                artifact_path=artifact,
                vocab_size=512,
                root=ROOT,
            )
            self.assertTrue(artifact.exists())
            self.assertEqual(tok.tokenizer_id, "magi_bringup_8k_v0.1")
            loaded = MagiByteBPETokenizer.load(artifact)
            self.assertEqual(loaded.vocab_size, tok.vocab_size)


if __name__ == "__main__":
    unittest.main()
