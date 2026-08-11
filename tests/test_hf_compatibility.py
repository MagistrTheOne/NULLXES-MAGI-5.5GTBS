#!/usr/bin/env python3
"""Real HuggingFace interoperability tests for MAGI."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

try:
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
except (ImportError, RuntimeError) as _HF_IMPORT_ERROR:
    torch = None  # type: ignore
    HF_AVAILABLE = False  # type: ignore
    _HF_IMPORT_ERROR = _HF_IMPORT_ERROR


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


@unittest.skipUnless(
    torch is not None and HF_AVAILABLE,
    "transformers+torch required for HF compatibility tests",
)
class TestHFCompatibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not HF_AVAILABLE or MagiConfig is None or MagiForCausalLM is None:
            raise unittest.SkipTest("transformers must be available for HF acceptance")

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

    def test_mask_equivalence_padded_batch(self):
        torch.manual_seed(21)
        model = MagiForCausalLM(native_config_to_hf(tiny_native_config()))
        model.eval()
        tokens = torch.tensor([[4, 5, 6, 7]], dtype=torch.long)
        with torch.no_grad():
            single = model(input_ids=tokens).logits[0]
            padded_ids = torch.tensor(
                [
                    [4, 5, 6, 7, 0, 0, 0, 0],
                    [0, 0, 0, 0, 4, 5, 6, 7],
                ],
                dtype=torch.long,
            )
            padded_mask = torch.tensor(
                [
                    [1, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 1],
                ],
                dtype=torch.long,
            )
            padded = model(input_ids=padded_ids, attention_mask=padded_mask).logits
        self.assertTrue(torch.allclose(single, padded[0, :4], atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(single, padded[1, 4:], atol=1e-5, rtol=1e-5))

    def test_kv_cache_continuity(self):
        torch.manual_seed(22)
        model = MagiForCausalLM(native_config_to_hf(tiny_native_config()))
        model.eval()
        prompt = torch.tensor([[3, 8, 9, 1]], dtype=torch.long)
        with torch.no_grad():
            full = model(input_ids=prompt, use_cache=False).logits
            step = model(input_ids=prompt[:, :2], use_cache=True)
            self.assertIsNotNone(step.past_key_values)
            self.assertGreater(step.past_key_values.get_seq_length(), 0)
            cont = model(
                input_ids=prompt[:, 2:],
                attention_mask=torch.ones((1, prompt.shape[1]), dtype=torch.long),
                past_key_values=step.past_key_values,
                use_cache=True,
            )
        self.assertTrue(torch.allclose(full[:, 2:], cont.logits, atol=1e-5, rtol=1e-5))

    def test_cached_vs_uncached_generation_logits(self):
        torch.manual_seed(23)
        cfg = native_config_to_hf(tiny_native_config())
        cfg.pad_token_id = 0
        cfg.eos_token_id = 1
        model = MagiForCausalLM(cfg)
        model.eval()
        prompt = torch.tensor([[2, 4, 6]], dtype=torch.long)
        with torch.no_grad():
            out_cache = model.generate(
                input_ids=prompt,
                max_new_tokens=3,
                do_sample=False,
                use_cache=True,
                return_dict_in_generate=True,
                output_scores=True,
            )
            out_no_cache = model.generate(
                input_ids=prompt,
                max_new_tokens=3,
                do_sample=False,
                use_cache=False,
                return_dict_in_generate=True,
                output_scores=True,
            )
        self.assertTrue(torch.equal(out_cache.sequences, out_no_cache.sequences))
        for score_a, score_b in zip(out_cache.scores, out_no_cache.scores):
            self.assertTrue(torch.allclose(score_a, score_b, atol=1e-5, rtol=1e-5))

    def test_output_attentions_real(self):
        model = MagiForCausalLM(native_config_to_hf(tiny_native_config()))
        model.eval()
        ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        with torch.no_grad():
            out = model(input_ids=ids, output_attentions=True, use_cache=False)
        self.assertIsNotNone(out.attentions)
        self.assertEqual(len(out.attentions), model.config.n_layers)
        self.assertEqual(out.attentions[0].shape[-1], ids.shape[1])

    def test_reorder_cache_beam(self):
        model = MagiForCausalLM(native_config_to_hf(tiny_native_config()))
        model.eval()
        ids = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.long)
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=True)
            original = out.past_key_values.to_legacy_cache()
            reordered = model._reorder_cache(out.past_key_values, torch.tensor([1, 0]))
            new_legacy = reordered.to_legacy_cache()
        self.assertEqual(len(new_legacy), model.config.n_layers)
        for layer_idx, (orig, new) in enumerate(zip(original, new_legacy)):
            self.assertTrue(torch.equal(new[0][0], orig[0][1]), layer_idx)
            self.assertTrue(torch.equal(new[0][1], orig[0][0]), layer_idx)

    def test_gradient_checkpointing_is_hard_disabled(self):
        model = MagiForCausalLM(native_config_to_hf(tiny_native_config()))
        self.assertFalse(model.supports_gradient_checkpointing)
        with self.assertRaises(RuntimeError):
            model.gradient_checkpointing_enable()

    def test_serialization_parity_native_hf_native(self):
        torch.manual_seed(24)
        native = MAGITransformer(tiny_native_config())
        hf = MagiForCausalLM(native_config_to_hf(tiny_native_config()))
        hf.model.load_state_dict(native.state_dict())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            hf.save_pretrained(path)
            loaded = AutoModelForCausalLM.from_pretrained(path)
            restored_native = MAGITransformer(tiny_native_config())
            restored_native.load_state_dict(hf_state_dict_to_native(loaded.state_dict()))
        for key, value in native.state_dict().items():
            self.assertTrue(torch.equal(value, restored_native.state_dict()[key]), key)

    def test_auto_from_pretrained_without_prior_magi_import(self):
        torch.manual_seed(25)
        cfg = native_config_to_hf(tiny_native_config())
        cfg.pad_token_id = 0
        model = MagiForCausalLM(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            model.save_pretrained(path)
            script = textwrap.dedent(
                f"""
                import sys
                sys.path.insert(0, {str(ROOT)!r})
                # Intentionally avoid `import magi` package root side effects.
                from transformers import AutoConfig, AutoModelForCausalLM
                cfg = AutoConfig.from_pretrained({str(path)!r}, trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained({str(path)!r}, trust_remote_code=True)
                assert cfg.model_type == "magi"
                assert "MagiForCausalLM" in model.__class__.__name__
                """
            )
            completed = subprocess.run(
                [sys.executable, "-c", script],
                check=False,
                capture_output=True,
                text=True,
            )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_hf_save_reload_state_dict_forward_loss_generate(self):
        torch.manual_seed(7)
        cfg = native_config_to_hf(tiny_native_config())
        cfg.bos_token_id = 0
        cfg.eos_token_id = 1
        cfg.pad_token_id = 0
        model = MagiForCausalLM(cfg)
        model.eval()
        input_ids = torch.tensor([[0, 2, 3, 4]], dtype=torch.long)
        labels = torch.tensor([[0, 2, 3, 4]], dtype=torch.long)
        output = model(input_ids=input_ids, labels=labels, output_hidden_states=True, use_cache=True)
        self.assertEqual(tuple(output.logits.shape), (1, 4, cfg.vocab_size))
        self.assertIsNotNone(output.loss)
        self.assertIsNotNone(output.hidden_states)
        self.assertIsNotNone(output.past_key_values)
        self.assertEqual(model.model.parameter_count(), sum(p.numel() for p in model.parameters()))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            model.save_pretrained(path)
            self.assertTrue((path / "config.json").exists())
            self.assertTrue((path / "generation_config.json").exists())
            self.assertTrue((path / "model.safetensors").exists() or (path / "pytorch_model.bin").exists())

            loaded_cfg = AutoConfig.from_pretrained(path)
            self.assertIsInstance(loaded_cfg, MagiConfig)
            loaded = AutoModelForCausalLM.from_pretrained(path)
            self.assertIsInstance(loaded, MagiForCausalLM)

            original_state = model.state_dict()
            loaded_state = loaded.state_dict()
            self.assertEqual(set(original_state), set(loaded_state))
            for key, value in original_state.items():
                self.assertTrue(torch.equal(value.cpu(), loaded_state[key].cpu()), key)

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
        out = model.get_output_embeddings()
        self.assertTrue(hasattr(out, "weight"))
        self.assertIs(out.weight, emb.weight)
        resized = model.resize_token_embeddings(80)
        self.assertEqual(resized.num_embeddings, 80)
        self.assertEqual(model.config.vocab_size, 80)
        model.tie_weights()
        self.assertIsNone(model.model.lm_head)


if __name__ == "__main__":
    unittest.main()
