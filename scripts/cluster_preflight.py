#!/usr/bin/env python3
"""Static cluster preflight summary for MAGI architecture configs."""

from __future__ import annotations

import argparse
from pathlib import Path

def main() -> int:
    parser = argparse.ArgumentParser(description="Print MAGI cluster preflight profiles")
    parser.add_argument("--profiles", type=Path, default=Path("configs/cluster_profiles_v0.1.yaml"))
    args = parser.parse_args()

    text = args.profiles.read_text(encoding="utf-8")
    training = _section_names(text, "training_profiles")
    serving = _section_names(text, "serving_profiles")
    print("training_profiles")
    for name in training:
        print(f"- {name}")
    print("serving_profiles")
    for name in serving:
        print(f"- {name}")
    return 0


def _section_names(text: str, section: str) -> list[str]:
    lines = text.splitlines()
    in_section = False
    names: list[str] = []
    for line in lines:
        if line.startswith(f"{section}:"):
            in_section = True
            continue
        if in_section and line and not line.startswith(" ") and line.endswith(":"):
            break
        if in_section and line.startswith("  ") and not line.startswith("    ") and line.strip().endswith(":"):
            names.append(line.strip()[:-1])
    return names


if __name__ == "__main__":
    raise SystemExit(main())
