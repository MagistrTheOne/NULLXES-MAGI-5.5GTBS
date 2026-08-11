#!/usr/bin/env python3
"""HF/native compatibility checks for MAGI production model configs."""

from __future__ import annotations

import unittest
from pathlib import Path

import torch

from magi.config import load_model_config
from magi.hf import MagiForCausalLM, native_config_to_hf
from magi.model import MAGITransformer

ROOT = Path(__file__).resolve().parents[1]

CONFIGS = [
    ROOT / "configs" / "magi_7b_moe_v0.1.yaml",
    ROOT / "configs" / "magi_7b_v0.1.yaml",
    ROOT / "configs" / "magi_casual_v0.1.yaml",
    ROOT / "configs" / "magi_35b_moe_v0.1.yaml",
    ROOT / "configs" / "magi_400b_v0.1.yaml",
]


class TestAllModelConfigsHF(unittest.TestCase):
    def test_all_configs_hf_meta_init_and_aliases(self):
        for path in CONFIGS:
            with self.subTest(config=path.name):
                cfg = load_model_config(path)
                hf = native_config_to_hf(cfg)
                self.assertEqual(hf.model_type, "magi")
                self.assertEqual(hf.num_hidden_layers, cfg.n_layers)
                self.assertEqual(hf.hidden_size, cfg.d_model)
                self.assertEqual(hf.num_attention_heads, cfg.n_heads)
                self.assertEqual(hf.num_key_value_heads, cfg.n_kv_heads)
                self.assertEqual(hf.head_dim, cfg.d_head)
                self.assertTrue(getattr(hf, "return_dict", True))
                with torch.device("meta"):
                    native = MAGITransformer.from_config(cfg)
                    model = MagiForCausalLM(hf)
                    model.tie_weights(recompute_mapping=False)
                self.assertEqual(native.parameter_count(), sum(p.numel() for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
