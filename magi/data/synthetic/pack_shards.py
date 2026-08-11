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

SHARD_VERSION = "v0.2"
TOKEN_DTYPE = "int32"
DEFAULT_TARGET_TOKENS_PER_SHARD = 65_536


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


def split_windows_by_token_budget(
    windows: Sequence[Sequence[int]],
    *,
    target_tokens_per_shard: int,
) -> list[list[Sequence[int]]]:
    if target_tokens_per_shard < 1:
        raise ValueError("target_tokens_per_shard must be >= 1")
    if not windows:
        raise ValueError("windows empty")
    shards: list[list[Sequence[int]]] = []
    current: list[Sequence[int]] = []
    current_tokens = 0
    for window in windows:
        wlen = len(window)
        if current and current_tokens + wlen > target_tokens_per_shard:
            shards.append(current)
            current = []
            current_tokens = 0
        current.append(window)
        current_tokens += wlen
    if current:
        shards.append(current)
    return shards


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
    raw_token_count: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    bin_path = output_dir / f"{shard_id}.bin"
    shard_hash = write_shard_bin(bin_path, windows)
    packed_token_count = sum(len(w) for w in windows)
    manifest = {
        "shard_id": shard_id,
        "version": SHARD_VERSION,
        "split": split,
        "tokenizer_id": tokenizer.tokenizer_id,
        "tokenizer_hash": tokenizer_hash,
        "sequence_length": len(windows[0]) if windows else 0,
        "raw_token_count": raw_token_count,
        "packed_token_count": packed_token_count,
        "training_token_count": packed_token_count,
        "token_count": packed_token_count,
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
            "generator_id": "magi_synth_templates_v0.2",
        },
        "shard_hash": shard_hash,
        "created_at": _utc_now(),
    }
    if config_hash is not None:
        manifest["config_hash"] = config_hash
    man_path = output_dir / f"{shard_id}.manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def write_training_shards(
    output_dir: Path,
    *,
    windows: Sequence[Sequence[int]],
    tokenizer: MagiByteBPETokenizer,
    tokenizer_hash: str,
    document_count: int,
    dataset_id: str,
    config_hash: str | None = None,
    split: str = "train",
    target_tokens_per_shard: int = DEFAULT_TARGET_TOKENS_PER_SHARD,
    raw_token_count: int | None = None,
) -> dict[str, Any]:
    chunks = split_windows_by_token_budget(
        windows, target_tokens_per_shard=target_tokens_per_shard
    )
    shard_manifests: list[dict[str, Any]] = []
    for i, chunk in enumerate(chunks):
        shard_id = f"{split}-{i:05d}"
        shard_manifests.append(
            write_training_shard(
                output_dir,
                shard_id=shard_id,
                windows=chunk,
                tokenizer=tokenizer,
                tokenizer_hash=tokenizer_hash,
                document_count=document_count if len(chunks) == 1 else 0,
                dataset_id=dataset_id,
                config_hash=config_hash,
                split=split,
                raw_token_count=raw_token_count if len(chunks) == 1 else None,
            )
        )
    total_packed = sum(int(m["packed_token_count"]) for m in shard_manifests)
    global_manifest = {
        "version": SHARD_VERSION,
        "dataset_id": dataset_id,
        "split": split,
        "tokenizer_id": tokenizer.tokenizer_id,
        "tokenizer_hash": tokenizer_hash,
        "target_tokens_per_shard": int(target_tokens_per_shard),
        "shard_count": len(shard_manifests),
        "shards": [m["shard_id"] for m in shard_manifests],
        "raw_token_count": raw_token_count,
        "packed_token_count": total_packed,
        "training_token_count": total_packed,
        "document_count": int(document_count),
        "created_at": _utc_now(),
    }
    if config_hash is not None:
        global_manifest["config_hash"] = config_hash
    (output_dir / "shards_manifest.json").write_text(
        json.dumps(global_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return global_manifest
