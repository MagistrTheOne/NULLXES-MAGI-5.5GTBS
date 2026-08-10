#!/usr/bin/env python3
"""stdlib unittest for MAGI production architecture inventory."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import spec_inventory  # noqa: E402


class TestSpecInventory(unittest.TestCase):
    def test_required_files_exist(self):
        _, missing = spec_inventory.inventory(ROOT)
        self.assertEqual(missing, [])


if __name__ == "__main__":
    unittest.main()
