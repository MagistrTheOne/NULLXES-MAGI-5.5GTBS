"""Byte-level BPE tokenizer for MAGI smoke gates.

stdlib-only. Production 131k candidates remain under tokenizer experiment specs.
"""

from __future__ import annotations

import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Iterable


SPECIAL_TOKENS = ("<unk>", "<pad>", "<bos>", "<eos>")


class MagiByteBPETokenizer:
    def __init__(
        self,
        *,
        merges: list[tuple[str, str]],
        vocab: dict[str, int],
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        bos_token: str = "<bos>",
        eos_token: str = "<eos>",
        normalization: str = "NFKC",
        tokenizer_id: str = "magi_t4_smoke_v0.1",
        version: str = "magi-tokenizer-v0.1",
    ) -> None:
        self.merges = [(a, b) for a, b in merges]
        self.vocab = dict(vocab)
        self.id_to_token = {idx: tok for tok, idx in self.vocab.items()}
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.normalization = normalization
        self.tokenizer_id = tokenizer_id
        self.version = version
        self.unk_id = self.vocab[unk_token]
        self.pad_id = self.vocab[pad_token]
        self.bos_id = self.vocab[bos_token]
        self.eos_id = self.vocab[eos_token]
        self._merge_ranks = {pair: i for i, pair in enumerate(self.merges)}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def normalize(self, text: str) -> str:
        if self.normalization:
            return unicodedata.normalize(self.normalization, text)
        return text

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        text = self.normalize(text)
        pieces = [chr(b) for b in text.encode("utf-8")]
        pieces = _apply_bpe(pieces, self._merge_ranks)
        ids = [self.vocab.get(piece, self.unk_id) for piece in pieces]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: Iterable[int], *, skip_special: bool = True) -> str:
        special = {self.unk_id, self.pad_id, self.bos_id, self.eos_id}
        raw: list[str] = []
        for token_id in ids:
            idx = int(token_id)
            if skip_special and idx in special:
                continue
            token = self.id_to_token.get(idx)
            if token is None or token in SPECIAL_TOKENS or token.startswith("<extra_"):
                continue
            raw.append(token)
        # Tokens are latin-1 code-units representing UTF-8 bytes.
        byte_values = bytes(ord(ch) & 0xFF for ch in "".join(raw))
        return byte_values.decode("utf-8", errors="replace")

    def to_dict(self) -> dict:
        return {
            "tokenizer_id": self.tokenizer_id,
            "version": self.version,
            "algorithm": "byte_bpe",
            "normalization": self.normalization,
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
            "vocab_size": self.vocab_size,
            "vocab": self.vocab,
            "merges": [list(pair) for pair in self.merges],
        }

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return output

    @classmethod
    def load(cls, path: str | Path) -> "MagiByteBPETokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        merges = [(a, b) for a, b in payload["merges"]]
        return cls(
            merges=merges,
            vocab={str(k): int(v) for k, v in payload["vocab"].items()},
            unk_token=payload.get("unk_token", "<unk>"),
            pad_token=payload.get("pad_token", "<pad>"),
            bos_token=payload.get("bos_token", "<bos>"),
            eos_token=payload.get("eos_token", "<eos>"),
            normalization=payload.get("normalization", "NFKC"),
            tokenizer_id=payload.get("tokenizer_id", "magi_tokenizer"),
            version=payload.get("version", "magi-tokenizer-v0.1"),
        )


def load_tokenizer(path: str | Path) -> MagiByteBPETokenizer:
    return MagiByteBPETokenizer.load(path)


def train_byte_bpe(
    texts: Iterable[str],
    *,
    vocab_size: int = 8192,
    min_merge_freq: int = 2,
    normalization: str = "NFKC",
    tokenizer_id: str = "magi_t4_smoke_v0.1",
) -> MagiByteBPETokenizer:
    if vocab_size < 260:
        raise ValueError("vocab_size must cover specials + 256 bytes")
    vocab: dict[str, int] = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
    for byte_value in range(256):
        vocab[chr(byte_value)] = len(vocab)

    corpus: list[list[str]] = []
    for text in texts:
        norm = unicodedata.normalize(normalization, text) if normalization else text
        if not norm:
            continue
        corpus.append([chr(b) for b in norm.encode("utf-8")])

    merges: list[tuple[str, str]] = []
    target_merges = vocab_size - len(vocab)
    while len(merges) < target_merges:
        pair_counts: Counter[tuple[str, str]] = Counter()
        for pieces in corpus:
            for i in range(len(pieces) - 1):
                pair_counts[(pieces[i], pieces[i + 1])] += 1
        if not pair_counts:
            break
        best_pair, best_freq = pair_counts.most_common(1)[0]
        if best_freq < min_merge_freq:
            break
        merges.append(best_pair)
        token = best_pair[0] + best_pair[1]
        vocab[token] = len(vocab)
        corpus = [_merge_pair(pieces, best_pair) for pieces in corpus]
        if len(vocab) >= vocab_size:
            break

    # Fill remaining ids deterministically if merges exhausted early.
    filler = 0
    while len(vocab) < vocab_size:
        name = f"<extra_{filler}>"
        vocab[name] = len(vocab)
        filler += 1

    return MagiByteBPETokenizer(
        merges=merges,
        vocab=vocab,
        normalization=normalization,
        tokenizer_id=tokenizer_id,
    )


def build_t4_smoke_tokenizer(
    *,
    seed_path: str | Path | None = None,
    artifact_path: str | Path | None = None,
    vocab_size: int = 8192,
) -> MagiByteBPETokenizer:
    root = Path(__file__).resolve().parents[2]
    seed = Path(seed_path) if seed_path else root / "tokenizer" / "data" / "t4_smoke_seed.txt"
    artifact = (
        Path(artifact_path)
        if artifact_path
        else root / "tokenizer" / "artifacts" / "magi_t4_smoke_v0.1.json"
    )
    if artifact.exists():
        return MagiByteBPETokenizer.load(artifact)
    texts = seed.read_text(encoding="utf-8").splitlines()
    tokenizer = train_byte_bpe(texts, vocab_size=vocab_size, tokenizer_id="magi_t4_smoke_v0.1")
    tokenizer.save(artifact)
    return tokenizer


def _merge_pair(pieces: list[str], pair: tuple[str, str]) -> list[str]:
    if len(pieces) < 2:
        return pieces
    out: list[str] = []
    i = 0
    a, b = pair
    merged = a + b
    while i < len(pieces):
        if i < len(pieces) - 1 and pieces[i] == a and pieces[i + 1] == b:
            out.append(merged)
            i += 2
        else:
            out.append(pieces[i])
            i += 1
    return out


def _apply_bpe(pieces: list[str], merge_ranks: dict[tuple[str, str], int]) -> list[str]:
    if len(pieces) < 2 or not merge_ranks:
        return pieces
    while True:
        pairs = [(pieces[i], pieces[i + 1]) for i in range(len(pieces) - 1)]
        ranked = [(merge_ranks[p], p) for p in pairs if p in merge_ranks]
        if not ranked:
            break
        _, best = min(ranked, key=lambda item: item[0])
        pieces = _merge_pair(pieces, best)
        if len(pieces) < 2:
            break
    return pieces
