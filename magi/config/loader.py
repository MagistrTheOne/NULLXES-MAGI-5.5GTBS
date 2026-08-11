"""Strict stdlib configuration loader for MAGI model configs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _strip_comment(line: str) -> str:
    in_single = False
    in_double = False
    out: list[str] = []
    for ch in line:
        if ch == "'" and not in_double:
            in_single = not in_single
            out.append(ch)
        elif ch == '"' and not in_single:
            in_double = not in_double
            out.append(ch)
        elif ch == "#" and not in_single and not in_double:
            break
        else:
            out.append(ch)
    return "".join(out).rstrip()


def _parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if value in {"", "null", "~"}:
        return None
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_simple_yaml(path: str | Path) -> dict[str, Any]:
    """Load the strict YAML subset used by MAGI configs.

    This intentionally avoids PyYAML so cluster preflight can run before Python
    dependencies are installed.
    """

    source = Path(path)
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for lineno, raw_line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        line = _strip_comment(raw_line)
        if not line.strip():
            continue
        if line.lstrip().startswith("- "):
            raise ValueError(f"{source}:{lineno}: list items are not supported in MAGI config YAML")
        if ":" not in line:
            raise ValueError(f"{source}:{lineno}: expected key: value")
        indent = len(line) - len(line.lstrip(" "))
        key, _, rest = line.strip().partition(":")
        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if rest.strip() == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(rest)
    return root


@dataclass(frozen=True)
class ModelConfig:
    name: str
    model_class: str
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_head: int
    vocab_size: int
    tied_embeddings: bool
    rmsnorm_eps: float
    rope_theta: float
    bias: bool
    n_dense_layers: int | None = None
    n_moe_layers: int | None = None
    d_ff: int | None = None
    d_ff_dense: int | None = None
    d_ff_expert: int | None = None
    n_routed_experts: int | None = None
    n_shared_experts: int | None = None
    top_k: int | None = None
    train_context: int | None = None
    infer_context: int | None = None
    # MoE runtime (from YAML `moe:`). Defaults keep smoke configs valid.
    moe_load_balance: str | None = None
    router_z_loss_coeff: float | None = None
    moe_bias_update_rate: float | None = None
    moe_gate: str | None = None

    @property
    def is_moe(self) -> bool:
        return self.model_class == "moe_decoder"

    @property
    def dense_ffn_dim(self) -> int:
        if self.d_ff is not None:
            return self.d_ff
        if self.d_ff_dense is not None:
            return self.d_ff_dense
        raise ValueError(f"{self.name}: dense FFN dimension is missing")

    def validate(self) -> None:
        if self.n_heads * self.d_head != self.d_model:
            raise ValueError(f"{self.name}: n_heads * d_head must equal d_model")
        if self.n_kv_heads <= 0 or self.n_heads % self.n_kv_heads != 0:
            raise ValueError(f"{self.name}: n_heads must be divisible by n_kv_heads")
        if self.is_moe:
            required = [
                self.n_dense_layers,
                self.n_moe_layers,
                self.d_ff_dense,
                self.d_ff_expert,
                self.n_routed_experts,
                self.n_shared_experts,
                self.top_k,
            ]
            if any(value is None for value in required):
                raise ValueError(f"{self.name}: MoE config is incomplete")
            if int(self.n_dense_layers) + int(self.n_moe_layers) != self.n_layers:
                raise ValueError(f"{self.name}: dense + MoE layers must equal n_layers")
            if int(self.top_k) > int(self.n_routed_experts):
                raise ValueError(f"{self.name}: top_k cannot exceed routed expert count")


def load_model_config(path: str | Path) -> ModelConfig:
    raw = load_simple_yaml(path)
    meta = raw.get("meta", {})
    arch = raw.get("architecture", {})
    moe = raw.get("moe", {}) or {}
    load_balance = moe.get("load_balance")
    z_coeff = moe.get("router_z_loss_coeff")
    bias_rate = moe.get("bias_update_rate", 1.0e-3 if load_balance == "aux_loss_free_bias" else 0.0)
    cfg = ModelConfig(
        name=str(meta["name"]),
        model_class=str(meta["model_class"]),
        d_model=int(arch["d_model"]),
        n_layers=int(arch["n_layers"]),
        n_heads=int(arch["n_heads"]),
        n_kv_heads=int(arch["n_kv_heads"]),
        d_head=int(arch["d_head"]),
        vocab_size=int(arch["vocab_size"]),
        tied_embeddings=bool(arch["tied_embeddings"]),
        rmsnorm_eps=float(arch["rmsnorm_eps"]),
        rope_theta=float(arch["rope_theta"]),
        bias=bool(arch["bias"]),
        n_dense_layers=_optional_int(arch.get("n_dense_layers")),
        n_moe_layers=_optional_int(arch.get("n_moe_layers")),
        d_ff=_optional_int(arch.get("d_ff")),
        d_ff_dense=_optional_int(arch.get("d_ff_dense")),
        d_ff_expert=_optional_int(arch.get("d_ff_expert")),
        n_routed_experts=_optional_int(arch.get("n_routed_experts")),
        n_shared_experts=_optional_int(arch.get("n_shared_experts")),
        top_k=_optional_int(arch.get("top_k")),
        train_context=_optional_int(arch.get("train_context")),
        infer_context=_optional_int(arch.get("infer_context")),
        moe_load_balance=None if load_balance is None else str(load_balance),
        router_z_loss_coeff=_optional_float(z_coeff),
        moe_bias_update_rate=_optional_float(bias_rate),
        moe_gate=None if moe.get("gate") is None else str(moe.get("gate")),
    )
    cfg.validate()
    return cfg


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)
