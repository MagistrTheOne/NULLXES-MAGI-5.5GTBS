#!/usr/bin/env python3
"""Real HuggingFace interoperability tests for MAGI."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch
import magi
from magi.config import ModelConfig, load_model_config
from magi.hf import (
    ARCHITECTURE_VERSION,
    CHECKPOINT_VERSION,
    CONFIG_VERSION,
    HF_AVAILABLE,
    MagiConfig,
    MagiForCausalLM,
    SERIALIZATION_VERSION,
    hf_config_to_native,
    hf_state_dict_to_native,
    native_config_to_hf,
    native_state_dict_to_hf,
)
from magi.hf.convert import save_native_yaml_from_hf
from magi.hf.serialization import load_state_dict, safetensors_available, save_state_dict
from magi.model import MAGITransformer
from transformers import AutoConfig, AutoModelForCausalLM


ROOT = Path(__file__).resolve().parents[1]


def tiny_native_config(*, tied: bool = True) -> ModelConfig:
    return ModelConfig(
        name="MAGI-HF-TINY",
        model_class="moe_decoder",
        d_model=32,
        n_layers=2,
        n_dense_layers=1,
        n_moe_layers=1,
        n_heads=4,
        n_kv_heads=2,
        d_head=8,
        vocab_size=64,
        tied_embeddings=tied,
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


class TestHFCompatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not HF_AVAILABLE or MagiConfig is None or MagiForCausalLM is None:
            raise RuntimeError("transformers must be available for Phase 0 HF acceptance")

    def test_auto_registration_is_active(self):
        self.assertTrue(magi.HF_AUTO_REGISTERED)
        cfg = AutoConfig.for_model("magi")
        self.assertIsInstance(cfg, MagiConfig)

    def test_native_load_path(self):
        cfg = load_model_config(ROOT / "configs" / "magi_35b_moe_v0.1.yaml")
        with torch.device("meta"):
            model = MAGITransformer.from_config(cfg)
        self.assertGreater(model.parameter_count(), 0)

    def test_config_roundtrip_and_versions(self):
        native = tiny_native_config()
        hf = native_config_to_hf(native)
        self.assertEqual(hf.architecture_version, ARCHITECTURE_VERSION)
        self.assertEqual(hf.checkpoint_version, CHECKPOINT_VERSION)
        self.assertEqual(hf.config_version, CONFIG_VERSION)
        self.assertEqual(hf.serialization_version, SERIALIZATION_VERSION)
        restored = hf_config_to_native(MagiConfig.from_dict(hf.to_dict()))
        self.assertEqual(native, restored)

    def test_native_yaml_hf_native_yaml_architecture(self):
        source = ROOT / "configs" / "magi_casual_v0.1.yaml"
        native = load_model_config(source)
        hf = MagiConfig.from_native_yaml(source)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "roundtrip.yaml"
            save_native_yaml_from_hf(hf, out)
            restored = load_model_config(out)
        self.assertEqual(native, restored)

    def test_state_dict_prefix_roundtrip(self):
        torch.manual_seed(3)
        native = MAGITransformer(tiny_native_config())
        hf_state = native_state_dict_to_hf(native.state_dict())
        back = hf_state_dict_to_native(hf_state)
        self.assertEqual(set(native.state_dict()), set(back))
        for key, value in native.state_dict().items():
            self.assertTrue(torch.equal(value, back[key]), key)

    def test_hf_save_reload_state_dict_forward_loss_generate(self):
        torch.manual_seed(7)
        cfg = native_config_to_hf(tiny_native_config())
        cfg.bos_token_id = 0
        cfg.eos_token_id = 1
        cfg.pad_token_id = 0
        model = MagiForCausalLM(cfg)
        input_ids = torch.tensor([[0, 2, 3, 4]], dtype=torch.long)
        labels = torch.tensor([[0, 2, 3, 4]], dtype=torch.long)
        output = model(input_ids=input_ids, labels=labels, output_hidden_states=True)
        self.assertEqual(tuple(output.logits.shape), (1, 4, cfg.vocab_size))
        self.assertIsNotNone(output.loss)
        self.assertIsNotNone(output.hidden_states)
        self.assertEqual(model.model.parameter_count(), sum(p.numel() for p in model.parameters()))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            model.save_pretrained(path)
            self.assertTrue((path / "config.json").exists())
            self.assertTrue((path / "generation_config.json").exists())
            self.assertTrue((path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists())

            loaded_cfg = AutoConfig.from_pretrained(path)
            self.assertIsInstance(loaded_cfg, MagiConfig)
            self.assertEqual(loaded_cfg.to_dict()["model_type"], "magi")
            loaded = AutoModelForCausalLM.from_pretrained(path)
            self.assertIsInstance(loaded, MagiForCausalLM)

            original_state = model.state_dict()
            loaded_state = loaded.state_dict()
            self.assertEqual(set(original_state), set(loaded_state))
            for key, value in original_state.items():
                self.assertTrue(torch.equal(value.cpu(), loaded_state[key].cpu()), key)

            self.assertEqual(
                hf_config_to_native(loaded_cfg),
                hf_config_to_native(cfg),
            )
            self.assertEqual(
                sum(p.numel() for p in model.parameters()),
                sum(p.numel() for p in loaded.parameters()),
            )

            reloaded_logits = loaded(input_ids=input_ids).logits
            self.assertTrue(torch.allclose(output.logits, reloaded_logits, atol=1e-5, rtol=1e-5))

            generated = loaded.generate(input_ids=input_ids[:, :2], max_new_tokens=2, do_sample=False)
            self.assertEqual(generated.shape[0], 1)
            self.assertEqual(generated.shape[1], 4)

    def test_cpu_init_and_dtype_move(self):
        cfg = native_config_to_hf(tiny_native_config(tied=False))
        model = MagiForCausalLM(cfg)
        model.to(device="cpu", dtype=torch.float32)
        input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
        logits = model(input_ids=input_ids).logits
        self.assertEqual(logits.device.type, "cpu")
        self.assertEqual(logits.dtype, torch.float32)

    def test_meta_device_init(self):
        cfg = native_config_to_hf(tiny_native_config())
        with torch.device("meta"):
            model = MagiForCausalLM(cfg)
        self.assertEqual(model.model.parameter_count(), sum(p.numel() for p in model.parameters()))

    def test_serialization_helpers_roundtrip(self):
        torch.manual_seed(11)
        native = MAGITransformer(tiny_native_config())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            written = save_state_dict(native.state_dict(), path, native=True)
            self.assertTrue(written.exists())
            if safetensors_available():
                self.assertEqual(written.name, "model.safetensors")
            loaded_native = load_state_dict(path, to_native=True)
            for key, value in native.state_dict().items():
                self.assertTrue(torch.equal(value, loaded_native[key]), key)

    def test_embedding_api(self):
        cfg = native_config_to_hf(tiny_native_config())
        model = MagiForCausalLM(cfg)
        emb = model.get_input_embeddings()
        self.assertEqual(emb.num_embeddings, cfg.vocab_size)
        resized = model.resize_token_embeddings(80)
        self.assertEqual(resized.num_embeddings, 80)
        self.assertEqual(model.config.vocab_size, 80)
        model.tie_weights()
        self.assertIsNone(model.model.lm_head)


if __name__ == "__main__":
    unittest.main()
