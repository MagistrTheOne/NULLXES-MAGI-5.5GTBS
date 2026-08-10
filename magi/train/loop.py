"""Single-GPU MAGI training loop for bring-up / T4 smoke."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Sequence

from magi.model.transformer import MAGITransformer
from magi.model.torch_runtime import require_torch
from magi.train.data import PackedTokenBatch
from magi.train.loss import causal_lm_loss, collect_router_entropy

torch = require_torch()


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 20
    lr: float = 1.0e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1.0e-8
    max_grad_norm: float = 1.0
    log_every: int = 1
    use_amp: bool = True
    seed: int = 42


@dataclass
class TrainMetrics:
    step: int
    loss: float
    grad_norm: float
    tokens_per_second: float
    router_entropy: float | None
    lr: float
    nan: bool


@dataclass
class TrainResult:
    history: list[TrainMetrics]
    optimizer: torch.optim.Optimizer
    scaler: Any
    summary: dict[str, Any]


def _grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for param in model.parameters():
        if param.grad is None:
            continue
        total += float(param.grad.detach().float().norm(2).item() ** 2)
    return math.sqrt(total)


def _make_scaler(enabled: bool, device_type: str):
    if not enabled or device_type != "cuda":
        return None
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda")
    return torch.cuda.amp.GradScaler()


def summarize_history(history: Sequence[TrainMetrics]) -> dict[str, Any]:
    if not history:
        raise ValueError("empty training history")
    if any(item.nan for item in history):
        return {
            "status": "NAN_STOP",
            "steps": len(history),
            "first_loss": history[0].loss,
            "last_loss": history[-1].loss,
        }
    first = history[0].loss
    last = history[-1].loss
    return {
        "status": "OK",
        "steps": len(history),
        "first_loss": first,
        "last_loss": last,
        "loss_delta": first - last,
        "loss_improved": last < first,
        "mean_tok_s": sum(item.tokens_per_second for item in history) / len(history),
        "final_grad_norm": history[-1].grad_norm,
        "final_router_entropy": history[-1].router_entropy,
    }


def train_steps(
    model: MAGITransformer,
    batches: Sequence[PackedTokenBatch],
    *,
    config: TrainConfig,
) -> TrainResult:
    if not batches:
        raise ValueError("batches must be non-empty")
    if config.steps < 1:
        raise ValueError("steps must be >= 1")

    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = next(model.parameters()).device
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.lr,
        betas=(config.beta1, config.beta2),
        eps=config.eps,
        weight_decay=config.weight_decay,
    )
    if config.use_amp and device.type == "cuda":
        for param in model.parameters():
            if param.dtype != torch.float32:
                raise ValueError(
                    "AMP GradScaler requires fp32 master weights; "
                    f"found parameter dtype={param.dtype}. "
                    "Move model with .to(device) only, not dtype=float16."
                )

    scaler = _make_scaler(config.use_amp and device.type == "cuda", device.type)
    history: list[TrainMetrics] = []

    for step in range(1, config.steps + 1):
        batch = batches[(step - 1) % len(batches)]
        optimizer.zero_grad(set_to_none=True)
        t0 = time.perf_counter()

        if scaler is not None:
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.float16)
        else:
            from contextlib import nullcontext

            autocast_ctx = nullcontext()

        with autocast_ctx:
            out = model(
                batch.input_ids,
                attention_mask=batch.attention_mask,
                use_cache=False,
                return_dict=True,
            )
            loss = causal_lm_loss(out.logits, batch.labels)

        if not torch.isfinite(loss):
            history.append(
                TrainMetrics(
                    step=step,
                    loss=float("nan"),
                    grad_norm=float("nan"),
                    tokens_per_second=0.0,
                    router_entropy=None,
                    lr=config.lr,
                    nan=True,
                )
            )
            break

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            grad_norm = _grad_norm(model)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            grad_norm = _grad_norm(model)
            optimizer.step()

        elapsed = max(time.perf_counter() - t0, 1.0e-9)
        tokens = int(batch.attention_mask.sum().item())
        entropy = collect_router_entropy(model)
        metrics = TrainMetrics(
            step=step,
            loss=float(loss.detach().float().item()),
            grad_norm=float(grad_norm),
            tokens_per_second=tokens / elapsed,
            router_entropy=None if entropy is None else float(entropy.float().item()),
            lr=config.lr,
            nan=False,
        )
        history.append(metrics)
        if config.log_every > 0 and (step % config.log_every == 0 or step == config.steps):
            ent = "n/a" if metrics.router_entropy is None else f"{metrics.router_entropy:.4f}"
            print(
                f"step={metrics.step} loss={metrics.loss:.6f} "
                f"grad_norm={metrics.grad_norm:.4f} tok/s={metrics.tokens_per_second:.1f} "
                f"router_entropy={ent}"
            )

    summary = summarize_history(history)
    return TrainResult(history=history, optimizer=optimizer, scaler=scaler, summary=summary)
