#!/usr/bin/env python3
"""stdlib unittest for param_count reconciliation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import param_count  # noqa: E402


class TestParamCount(unittest.TestCase):
    def test_magi_35b_moe(self):
        r = param_count.analyze(ROOT / "configs" / "magi_35b_moe_v0.1.yaml")
        self.assertEqual(r["total"], 34_054_674_432)
        self.assertEqual(r["active_per_token"], 8_335_202_304)
        self.assertGreaterEqual(r["total"], 30_000_000_000)
        self.assertLessEqual(r["total"], 40_000_000_000)
        self.assertEqual(r["active_experts_per_moe_layer"], 5)

    def test_magi_400b(self):
        r = param_count.analyze(ROOT / "configs" / "magi_400b_v0.1.yaml")
        self.assertEqual(r["total"], 406_626_246_656)
        self.assertEqual(r["active_per_token"], 43_164_639_232)
        self.assertGreaterEqual(r["total"], 400_000_000_000)
        self.assertEqual(r["active_experts_per_moe_layer"], 10)

    def test_magi_casual(self):
        r = param_count.analyze(ROOT / "configs" / "magi_casual_v0.1.yaml")
        self.assertEqual(r["total"], 13_789_271_040)
        self.assertEqual(r["active_per_token"], r["total"])
        self.assertEqual(r["components"]["lm_head"], 0)


if __name__ == "__main__":
    unittest.main()
