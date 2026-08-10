#!/usr/bin/env python3
"""Tests for MAGI native model core."""

from __future__ import annotations

import unittest

from magi.config import ModelConfig


try:
    import torch
    from magi.model import MAGITransformer
except (ImportError, RuntimeError):
    torch = None
    MAGITransformer = None


class TestMAGIModelCore(unittest.TestCase):
    def test_dense_forward_shape_when_torch_available(self):
        if torch is None or MAGITransformer is None:
            self.skipTest("PyTorch is not installed")
        cfg = ModelConfig(
            name="MAGI-TINY-DENSE",
            model_class="dense_decoder",
            d_model=32,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            d_head=8,
            vocab_size=64,
            tied_embeddings=True,
            rmsnorm_eps=1.0e-6,
            rope_theta=10000.0,
            bias=False,
            d_ff=64,
            train_context=16,
            infer_context=16,
        )
        model = MAGITransformer(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 5), dtype=torch.long)
        logits = model(input_ids)
        self.assertEqual(tuple(logits.shape), (2, 5, cfg.vocab_size))

    def test_moe_forward_shape_when_torch_available(self):
        if torch is None or MAGITransformer is None:
            self.skipTest("PyTorch is not installed")
        cfg = ModelConfig(
            name="MAGI-TINY-MOE",
            model_class="moe_decoder",
            d_model=32,
            n_layers=2,
            n_dense_layers=1,
            n_moe_layers=1,
            n_heads=4,
            n_kv_heads=2,
            d_head=8,
            vocab_size=64,
            tied_embeddings=False,
            rmsnorm_eps=1.0e-6,
            rope_theta=10000.0,
            bias=False,
            d_ff_dense=64,
            d_ff_expert=16,
            n_routed_experts=4,
            n_shared_experts=1,
            top_k=2,
            train_context=16,
            infer_context=16,
        )
        model = MAGITransformer(cfg)
        input_ids = torch.randint(0, cfg.vocab_size, (2, 5), dtype=torch.long)
        logits = model(input_ids)
        self.assertEqual(tuple(logits.shape), (2, 5, cfg.vocab_size))


if __name__ == "__main__":
    unittest.main()
