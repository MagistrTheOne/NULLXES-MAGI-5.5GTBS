"""Token packing into MAGI training shards (.bin + JSON manifest)."""

from __future__ import annotations

import json
import struct
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Sequence

from magi.data.synthetic.record import SyntheticRecord
from magi.tokenizer.byte_bpe import MagiByteBPETokenizer

SHARD_VERSION = "v0.1"
TOKEN_DTYPE = "int32"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def tokenize_records(
    tokenizer: MagiByteBPETokenizer,
    records: Sequence[SyntheticRecord],
) -> list[int]:
    stream: list[int] = []
    for rec in records:
        stream.extend(tokenizer.encode(rec.text, add_bos=True, add_eos=True))
    return stream


def pack_token_windows(token_ids: Sequence[int], *, seq_len: int) -> list[list[int]]:
    if seq_len < 2:
        raise ValueError("seq_len must be >= 2")
    if not token_ids:
        raise ValueError("token_ids empty")
    stream = list(token_ids)
    if len(stream) < seq_len:
        while len(stream) < seq_len:
            stream.extend(token_ids)
    usable = (len(stream) // seq_len) * seq_len
    stream = stream[:usable]
    return [stream[i : i + seq_len] for i in range(0, usable, seq_len)]


def write_shard_bin(path: Path, windows: Sequence[Sequence[int]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = sha256()
    with path.open("wb") as handle:
        handle.write(struct.pack("<I", len(windows)))
        handle.write(struct.pack("<I", len(windows[0]) if windows else 0))
        for window in windows:
            raw = struct.pack(f"<{len(window)}i", *[int(x) for x in window])
            handle.write(raw)
            digest.update(raw)
    return digest.hexdigest()


def read_shard_bin(path: Path) -> list[list[int]]:
    data = Path(path).read_bytes()
    if len(data) < 8:
        raise ValueError(f"shard too small: {path}")
    n_windows, seq_len = struct.unpack_from("<II", data, 0)
    offset = 8
    windows: list[list[int]] = []
    for _ in range(n_windows):
        need = seq_len * 4
        chunk = data[offset : offset + need]
        if len(chunk) != need:
            raise ValueError(f"truncated shard window in {path}")
        windows.append(list(struct.unpack(f"<{seq_len}i", chunk)))
        offset += need
    return windows


def write_training_shard(
    output_dir: Path,
    *,
    shard_id: str,
    windows: Sequence[Sequence[int]],
    tokenizer: MagiByteBPETokenizer,
    tokenizer_hash: str,
    document_count: int,
    dataset_id: str,
    config_hash: str | None = None,
    split: str = "train",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    bin_path = output_dir / f"{shard_id}.bin"
    shard_hash = write_shard_bin(bin_path, windows)
    token_count = sum(len(w) for w in windows)
    manifest = {
        "shard_id": shard_id,
        "version": SHARD_VERSION,
        "split": split,
        "tokenizer_id": tokenizer.tokenizer_id,
        "tokenizer_hash": tokenizer_hash,
        "sequence_length": len(windows[0]) if windows else 0,
        "token_count": token_count,
        "document_count": int(document_count),
        "dataset_lineage": [
            {
                "dataset_id": dataset_id,
                "weight": 1.0,
                "license_status": "NULLXES_SYNTHETIC",
            }
        ],
        "packing": {
            "document_boundaries_preserved": True,
            "eos_between_documents": True,
            "pad_policy": "none_windowed",
            "token_dtype": TOKEN_DTYPE,
            "bin_file": bin_path.name,
        },
        "quality_summary": {
            "windows": len(windows),
            "generator_id": "magi_synth_templates_v0.1",
        },
        "shard_hash": shard_hash,
        "created_at": _utc_now(),
    }
    if config_hash is not None:
        manifest["config_hash"] = config_hash
    man_path = output_dir / f"{shard_id}.manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
