"""T4 training smoke unit tests (CPU)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
    import torch
    from magi.config import load_model_config
    from magi.model import MAGITransformer
    from magi.tokenizer import build_t4_smoke_tokenizer
    from magi.train import TrainConfig, pack_texts, save_train_checkpoint, train_steps
    from magi.train.data import load_corpus_lines
except (ImportError, RuntimeError):
    torch = None  # type: ignore


@unittest.skipIf(torch is None, "torch unavailable")
class T4TrainSmokeTests(unittest.TestCase):
    def test_train_steps_improve_loss_and_checkpoint(self):
        cfg = load_model_config(ROOT / "configs" / "magi_t4_smoke_v0.1.yaml")
        tokenizer = build_t4_smoke_tokenizer(vocab_size=cfg.vocab_size)
        texts = load_corpus_lines(ROOT / "tokenizer" / "data" / "t4_smoke_seed.txt")
        model = MAGITransformer.from_config(cfg)
        batches = pack_texts(tokenizer, texts, seq_len=64, batch_size=1, device="cpu")
        result = train_steps(
            model,
            batches,
            config=TrainConfig(steps=5, lr=3.0e-4, use_amp=False, seed=7, log_every=0),
        )
        self.assertEqual(result.summary["status"], "OK")
        self.assertTrue(result.summary["loss_improved"])
        self.assertGreater(result.summary["first_loss"], result.summary["last_loss"])
        self.assertIsNotNone(result.history[-1].router_entropy)

        with tempfile.TemporaryDirectory() as tmp:
            path = save_train_checkpoint(
                tmp,
                model=model,
                optimizer=result.optimizer,
                scaler=None,
                step=result.history[-1].step,
                loss=result.history[-1].loss,
                config_path=ROOT / "configs" / "magi_t4_smoke_v0.1.yaml",
                model_name=cfg.name,
                tokenizer_id=tokenizer.tokenizer_id,
                tokenizer_sha256="test",
                metrics=result.summary,
            )
            self.assertTrue(path.exists())
            self.assertEqual(path.name, "model.safetensors")
            self.assertTrue((Path(tmp) / "model.safetensors").exists())
            self.assertTrue((Path(tmp) / "train_meta.json").exists())


if __name__ == "__main__":
    unittest.main()
