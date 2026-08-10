"""Synthetic MAGI corpus tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from magi.data.synthetic import GENERATOR_ID, build_synthetic_dataset, generate_records, validate_pins
from magi.data.synthetic.pack_shards import read_shard_bin
from magi.data.synthetic.record import GENERATOR_LICENSE


class SyntheticDatasetTests(unittest.TestCase):
    def test_pins_and_determinism(self):
        a = generate_records(n_docs=32, seed=42)
        b = generate_records(n_docs=32, seed=42)
        self.assertEqual([r.id for r in a], [r.id for r in b])
        self.assertEqual([r.text for r in a], [r.text for r in b])
        for rec in a:
            self.assertEqual(rec.generator_id, GENERATOR_ID)
            self.assertEqual(rec.license, GENERATOR_LICENSE)
            validate_pins(rec.text, rec.semantic_pins, record_id=rec.id)

    def test_build_writes_manifests_and_shard_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = build_synthetic_dataset(
                output_dir=tmp,
                n_docs=64,
                seed=7,
                seq_len=64,
                tokenizer_vocab=8192,
                write_shards=True,
                config_path=ROOT / "configs" / "synthetic_magi_v0.1.yaml",
            )
            self.assertEqual(report["status"], "OK")
            self.assertEqual(report["n_docs"], 64)
            self.assertEqual(report["pin_pass_rate"], 1.0)

            out = Path(tmp)
            for name in (
                "records.jsonl",
                "dataset_manifest.json",
                "generator_manifest.json",
                "contamination_report.json",
                "build_report.json",
                "shards/train-00000.bin",
                "shards/train-00000.manifest.json",
            ):
                self.assertTrue((out / name).exists(), name)

            manifest = json.loads((out / "dataset_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["source_type"], "nullxes_synthetic")
            self.assertEqual(manifest["license_status"], GENERATOR_LICENSE)
            self.assertTrue(manifest["synthetic"]["is_synthetic"])
            self.assertEqual(manifest["synthetic"]["generator_id"], GENERATOR_ID)

            shard_man = json.loads((out / "shards" / "train-00000.manifest.json").read_text(encoding="utf-8"))
            windows = read_shard_bin(out / "shards" / "train-00000.bin")
            self.assertEqual(len(windows) * shard_man["sequence_length"], shard_man["token_count"])
            self.assertEqual(report["shard_token_count"], shard_man["token_count"])

    def test_golden_path_exists_when_committed(self):
        golden = ROOT / "data" / "synthetic" / "magi_synth_v0.1" / "golden" / "records.jsonl"
        if not golden.exists():
            self.skipTest("golden sample not generated yet")
        lines = [line for line in golden.read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertGreaterEqual(len(lines), 32)
        first = json.loads(lines[0])
        self.assertIn("semantic_pins", first)
        self.assertEqual(first["license"], GENERATOR_LICENSE)


if __name__ == "__main__":
    unittest.main()
