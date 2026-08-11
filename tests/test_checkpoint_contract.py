"""Checkpoint contract + MoE expert_bias resume acceptance."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MICRO_MOE_YAML = """
meta:
  name: MAGI-UNIT-MOE
  version: v0.1
  program_parent: MAGI-5.5GTBS
  status: unit_fixture
  model_class: moe_decoder
  claim_class_topology: CALCULATED

architecture:
  type: decoder_only_prenorm_sparse_moe
  d_model: 64
  n_layers: 2
  n_dense_layers: 1
  n_moe_layers: 1
  n_heads: 4
  n_kv_heads: 2
  d_head: 16
  d_ff_dense: 128
  d_ff_expert: 64
  n_routed_experts: 4
  n_shared_experts: 1
  top_k: 2
  vocab_size: 256
  tied_embeddings: true
  norm: rmsnorm
  rmsnorm_eps: 1.0e-6
  activation: swiglu
  positional: rope
  rope_theta: 10000.0
  bias: false
  attention: gqa
  train_context: 64
  infer_context: 64

moe:
  gate: sigmoid_normalize_topk
  gate_dtype: fp32
  load_balance: aux_loss_free_bias
  router_z_loss_coeff: 1.0e-5
  bias_update_rate: 1.0e-3

tokenizer:
  id: magi_unit_fixture
  vocab_size: 256
"""

try:
    import torch
    from magi.checkpoint.manifest import MANIFEST_VERSION, file_sha256
    from magi.config import load_model_config
    from magi.model import MAGITransformer
    from magi.train import TrainConfig, train_steps
    from magi.train.checkpoint import (
        CHECKPOINT_FORMAT,
        capture_rng_state,
        load_train_checkpoint,
        restore_rng_state,
        save_train_checkpoint,
    )
    from magi.train.data import PackedTokenBatch
except (ImportError, RuntimeError):
    torch = None  # type: ignore


def _expert_bias_tensors(model: torch.nn.Module) -> list[torch.Tensor]:
    return [
        buf.detach().cpu().clone()
        for name, buf in model.named_buffers()
        if name.endswith("expert_bias")
    ]


def _synthetic_batches(vocab_size: int, *, seq_len: int = 16, n: int = 2) -> list[PackedTokenBatch]:
    batches: list[PackedTokenBatch] = []
    for i in range(n):
        torch.manual_seed(100 + i)
        ids = torch.randint(0, vocab_size, (1, seq_len), dtype=torch.long)
        labels = ids.clone()
        batches.append(PackedTokenBatch(input_ids=ids, attention_mask=None, labels=labels))
    return batches


@unittest.skipIf(torch is None, "torch unavailable")
class CheckpointContractTests(unittest.TestCase):
    def test_manifest_inventory_and_expert_bias_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "unit_moe.yaml"
            cfg_path.write_text(MICRO_MOE_YAML, encoding="utf-8")
            cfg = load_model_config(cfg_path)
            self.assertTrue(cfg.is_moe)
            batches = _synthetic_batches(cfg.vocab_size)
            model = MAGITransformer.from_config(cfg)

            result = train_steps(
                model,
                batches,
                config=TrainConfig(steps=3, lr=3.0e-4, use_amp=False, seed=11, log_every=0),
            )
            self.assertEqual(result.summary["status"], "OK")
            biases_before = _expert_bias_tensors(model)
            self.assertTrue(biases_before, "expected MoE expert_bias buffers")
            if all(torch.equal(b, torch.zeros_like(b)) for b in biases_before):
                for name, buf in model.named_buffers():
                    if name.endswith("expert_bias"):
                        buf.add_(
                            torch.linspace(-0.1, 0.1, buf.numel(), device=buf.device).view_as(buf)
                        )
                biases_before = _expert_bias_tensors(model)

            data_manifest = ROOT / "data" / "program" / "MAGI_DATA_MANIFEST_v0.1.yaml"
            out = Path(tmp) / "ckpt"
            path = save_train_checkpoint(
                out,
                model=model,
                optimizer=result.optimizer,
                scaler=None,
                step=result.history[-1].step,
                loss=result.history[-1].loss,
                config_path=cfg_path,
                model_name=cfg.name,
                tokenizer_id="magi_unit_fixture",
                tokenizer_sha256="test-tok-sha",
                metrics=result.summary,
                save_optimizer=True,
                save_rng=True,
                dataset_manifest_id="MAGI_DATA_MANIFEST_v0.1",
                dataset_manifest_sha256=file_sha256(data_manifest) if data_manifest.exists() else "UNBOUND",
                mixture_id="base_mixture_v0_1",
                train_config={"lr": 3.0e-4, "steps": 3, "seq_len": 16, "batch_size": 1},
                consumed_tokens=int(result.summary["consumed_tokens"]),
                run_id="acceptance-expert-bias",
            )
            self.assertTrue(path.exists())
            step_dir = out / f"step-{result.history[-1].step:06d}"
            meta = json.loads((step_dir / "train_meta.json").read_text(encoding="utf-8"))
            manifest = meta["manifest"]

            self.assertEqual(manifest["manifest_version"], MANIFEST_VERSION)
            self.assertEqual(manifest["checkpoint_format"], CHECKPOINT_FORMAT)
            self.assertEqual(manifest["checkpoint_schema_version"], "1")
            self.assertEqual(manifest["global_step"], result.history[-1].step)
            self.assertEqual(manifest["consumed_tokens"], result.summary["consumed_tokens"])
            self.assertIn("model", manifest["artifacts"])
            self.assertIn("optimizer", manifest["artifacts"])
            self.assertIn("rng", manifest["artifacts"])

            for key, art in manifest["artifacts"].items():
                disk = step_dir / art["path"]
                self.assertTrue(disk.is_file(), key)
                self.assertEqual(file_sha256(disk), art["sha256"], key)

            rng_snap = capture_rng_state()
            live_opt = copy.deepcopy(result.optimizer.state_dict())
            ref_model = MAGITransformer.from_config(cfg)
            ref_model.load_state_dict(model.state_dict())
            ref_opt = torch.optim.AdamW(ref_model.parameters(), lr=3.0e-4)
            ref_opt.load_state_dict(live_opt)
            restore_rng_state(rng_snap)
            ref = train_steps(
                ref_model,
                batches,
                config=TrainConfig(steps=1, lr=3.0e-4, use_amp=False, seed=11, log_every=0),
                optimizer=ref_opt,
                start_step=result.history[-1].step,
                consumed_tokens=int(result.summary["consumed_tokens"]),
                reseed=False,
            )

            fresh = MAGITransformer.from_config(cfg)
            opt = torch.optim.AdamW(fresh.parameters(), lr=3.0e-4)
            loaded = load_train_checkpoint(
                step_dir,
                model=fresh,
                optimizer=opt,
                restore_rng=True,
            )
            self.assertEqual(loaded["step"], result.history[-1].step)
            for a, b in zip(biases_before, _expert_bias_tensors(fresh)):
                self.assertTrue(torch.equal(a, b), "expert_bias must round-trip exactly")

            cont = train_steps(
                fresh,
                batches,
                config=TrainConfig(steps=1, lr=3.0e-4, use_amp=False, seed=11, log_every=0),
                optimizer=opt,
                start_step=int(loaded["step"]),
                consumed_tokens=int(loaded.get("consumed_tokens") or 0),
                reseed=False,
            )
            self.assertEqual(cont.summary["status"], "OK")
            self.assertEqual(ref.summary["status"], "OK")
            self.assertAlmostEqual(cont.history[-1].loss, ref.history[-1].loss, places=5)


if __name__ == "__main__":
    unittest.main()
