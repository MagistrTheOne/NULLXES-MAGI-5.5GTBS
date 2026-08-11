"""Single-GPU training loops for MAGI bring-up and hardware smokes."""

from magi.train.checkpoint import (
    TrainCheckpoint,
    capture_rng_state,
    load_train_checkpoint,
    restore_rng_state,
    save_train_checkpoint,
)
from magi.train.data import PackedTokenBatch, load_jsonl_texts, load_shard_batches, pack_texts
from magi.train.loop import TrainConfig, TrainMetrics, TrainResult, train_steps
from magi.train.loss import causal_lm_loss, magi_train_loss

__all__ = [
    "PackedTokenBatch",
    "TrainCheckpoint",
    "TrainConfig",
    "TrainMetrics",
    "TrainResult",
    "capture_rng_state",
    "causal_lm_loss",
    "load_jsonl_texts",
    "load_shard_batches",
    "load_train_checkpoint",
    "magi_train_loss",
    "pack_texts",
    "restore_rng_state",
    "save_train_checkpoint",
    "train_steps",
]
