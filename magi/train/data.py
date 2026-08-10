"""Token packing for MAGI single-GPU training smokes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from magi.model.torch_runtime import require_torch
from magi.tokenizer.byte_bpe import MagiByteBPETokenizer

torch = require_torch()


@dataclass(frozen=True)
class PackedTokenBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


def load_corpus_lines(path: str | Path) -> list[str]:
    """Load plain-text lines or synthetic records.jsonl texts."""
    target = Path(path)
    if target.suffix.lower() == ".jsonl":
        return load_jsonl_texts(target)
    text = target.read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"corpus is empty: {path}")
    return lines


def load_jsonl_texts(path: str | Path) -> list[str]:
    texts: list[str] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        text = str(raw.get("text", "")).strip()
        if not text:
            raise ValueError(f"{path}:{line_no}: missing text")
        texts.append(text)
    if not texts:
        raise ValueError(f"corpus is empty: {path}")
    return texts


def pack_token_ids(
    token_windows: Sequence[Sequence[int]],
    *,
    batch_size: int,
    device: torch.device | str,
    pad_id: int,
) -> list[PackedTokenBatch]:
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if not token_windows:
        raise ValueError("token_windows empty")
    batches: list[PackedTokenBatch] = []
    windows = [list(w) for w in token_windows]
    for start in range(0, len(windows), batch_size):
        chunk = windows[start : start + batch_size]
        while len(chunk) < batch_size:
            chunk.append(windows[0])
        ids = torch.tensor(chunk, dtype=torch.long, device=device)
        attention_mask = (ids != pad_id).long()
        labels = ids.clone()
        labels[attention_mask == 0] = -100
        batches.append(PackedTokenBatch(input_ids=ids, attention_mask=attention_mask, labels=labels))
    return batches


def load_shard_batches(
    shard_bin: str | Path,
    *,
    batch_size: int,
    device: torch.device | str,
    pad_id: int,
) -> list[PackedTokenBatch]:
    from magi.data.synthetic.pack_shards import read_shard_bin

    windows = read_shard_bin(Path(shard_bin))
    return pack_token_ids(windows, batch_size=batch_size, device=device, pad_id=pad_id)


def pack_texts(
    tokenizer: MagiByteBPETokenizer,
    texts: Sequence[str],
    *,
    seq_len: int,
    batch_size: int,
    device: torch.device | str,
    pad_id: int | None = None,
) -> list[PackedTokenBatch]:
    if seq_len < 2:
        raise ValueError("seq_len must be >= 2 for causal LM")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    pad = tokenizer.pad_id if pad_id is None else int(pad_id)

    stream: list[int] = []
    for text in texts:
        stream.extend(tokenizer.encode(text, add_bos=True, add_eos=True))
    if len(stream) < seq_len:
        # Repeat corpus until one full window exists.
        while len(stream) < seq_len:
            stream.extend(stream)
    stream = stream[: (len(stream) // seq_len) * seq_len]
    if not stream:
        raise ValueError("failed to pack token stream")

    windows = [stream[i : i + seq_len] for i in range(0, len(stream), seq_len)]
    batches: list[PackedTokenBatch] = []
    for start in range(0, len(windows), batch_size):
        chunk = windows[start : start + batch_size]
        if len(chunk) < batch_size:
            # Pad last incomplete batch by repeating first window.
            while len(chunk) < batch_size:
                chunk.append(windows[0])
        ids = torch.tensor(chunk, dtype=torch.long, device=device)
        attention_mask = (ids != pad).long()
        labels = ids.clone()
        labels[attention_mask == 0] = -100
        batches.append(PackedTokenBatch(input_ids=ids, attention_mask=attention_mask, labels=labels))
    return batches
