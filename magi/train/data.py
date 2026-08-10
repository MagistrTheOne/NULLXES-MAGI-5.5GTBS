"""Token packing for MAGI single-GPU training smokes."""

from __future__ import annotations

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
    text = Path(path).read_text(encoding="utf-8")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"corpus is empty: {path}")
    return lines


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
