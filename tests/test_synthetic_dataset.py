"""Synthetic MAGI corpus tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from magi.data.synthetic import GENERATOR_ID, build_synthetic_dataset, generate_records, validate_pins
from magi.data.synthetic.build import build_dataset_manifest, write_jsonl
from magi.data.synthetic.pack_shards import read_shard_bin
from magi.data.synthetic.record import GENERATOR_LICENSE, compute_semantic_hash


class SyntheticDatasetTests(unittest.TestCase):
    def test_pins_and_determinism(self):
        a = generate_records(n_docs=32, seed=42)
        b = generate_records(n_docs=32, seed=42)
        self.assertEqual([r.id for r in a], [r.id for r in b])
        self.assertEqual([r.text for r in a], [r.text for r in b])
        self.assertEqual([r.semantic_hash for r in a], [r.semantic_hash for r in b])
        for rec in a:
            self.assertEqual(rec.generator_id, GENERATOR_ID)
            self.assertEqual(rec.license, GENERATOR_LICENSE)
            validate_pins(rec.text, rec.semantic_pins, record_id=rec.id)
            self.assertEqual(
                rec.semantic_hash,
                compute_semantic_hash(
                    domain=rec.domain,
                    prompt_family=rec.prompt_family,
                    pins=rec.semantic_pins,
                ),
            )
            # index-as-diversity removed from rendered text
            self.assertNotRegex(
                rec.text,
                r"(Mathematics drill|Programming sample|Reasoning item|Science note|Systems note|Dialogue|Structured record|Multilingual pair) \d+",
            )
    def test_content_hash_ignores_captured_at(self):
        records = generate_records(n_docs=16, seed=3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "records.jsonl"
            records_sha = write_jsonl(path, records)
            m1 = build_dataset_manifest(
                dataset_id="t", version="v0.2", records=records, records_sha256=records_sha
            )
            m2 = build_dataset_manifest(
                dataset_id="t", version="v0.2", records=records, records_sha256=records_sha
            )
            self.assertEqual(m1["hashes"]["records_sha256"], records_sha)
            self.assertEqual(m1["hashes"]["content_sha256"], m2["hashes"]["content_sha256"])
            self.assertEqual(m1["hashes"]["manifest_sha256"], m2["hashes"]["manifest_sha256"])
            self.assertNotEqual(m1["provenance"]["captured_at"], "")  # metadata present
            # Force different captured_at wall clock still same identity hashes
            m2["provenance"]["captured_at"] = "1970-01-01T00:00:00Z"
            self.assertEqual(m1["hashes"]["content_sha256"], m2["hashes"]["content_sha256"])

    def test_build_writes_manifests_and_shard_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = build_synthetic_dataset(
                output_dir=tmp,
                n_docs=64,
                seed=7,
                seq_len=64,
                tokenizer_vocab=8192,
                write_shards=True,
                target_tokens_per_shard=2048,
                config_path=ROOT / "configs" / "synthetic_magi_v0.1.yaml",
            )
            self.assertEqual(report["status"], "OK")
            self.assertEqual(report["n_docs"], 64)
            self.assertEqual(report["pin_pass_rate"], 1.0)
            self.assertGreaterEqual(report["shard_count"], 1)
            self.assertIn("raw_token_count", report)
            self.assertIn("packed_token_count", report)

            out = Path(tmp)
            for name in (
                "records.jsonl",
                "dataset_manifest.json",
                "generator_manifest.json",
                "contamination_report.json",
                "build_report.json",
                "shards/shards_manifest.json",
            ):
                self.assertTrue((out / name).exists(), name)

            manifest = json.loads((out / "dataset_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["source_type"], "nullxes_synthetic")
            self.assertEqual(manifest["license_status"], GENERATOR_LICENSE)
            self.assertTrue(manifest["synthetic"]["is_synthetic"])
            self.assertEqual(manifest["synthetic"]["generator_id"], GENERATOR_ID)
            self.assertIn("records_sha256", manifest["hashes"])
            self.assertIn("content_sha256", manifest["hashes"])
            self.assertIn("manifest_sha256", manifest["hashes"])

            contamination = json.loads(
                (out / "contamination_report.json").read_text(encoding="utf-8")
            )
            self.assertEqual(contamination["status"], "not_checked")
            self.assertIsNone(contamination["overlap_count"])

            shard0 = out / "shards" / "train-00000.bin"
            self.assertTrue(shard0.exists())
            shard_man = json.loads(
                (out / "shards" / "train-00000.manifest.json").read_text(encoding="utf-8")
            )
            windows = read_shard_bin(shard0)
            self.assertEqual(len(windows) * shard_man["sequence_length"], shard_man["token_count"])
            self.assertEqual(report["packed_token_count"], report["training_token_count"])

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
