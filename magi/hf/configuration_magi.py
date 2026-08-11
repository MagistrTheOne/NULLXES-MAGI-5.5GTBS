"""HuggingFace configuration for MAGI."""

from __future__ import annotations

from pathlib import Path

from magi.config import ModelConfig, load_model_config, load_simple_yaml

try:
    from magi.hf.versions import (
        ARCHITECTURE_VERSION,
        CHECKPOINT_VERSION,
        CONFIG_VERSION,
        SERIALIZATION_VERSION,
    )
except ImportError:  # checkpoint remote-code layout
    from versions import (  # type: ignore
        ARCHITECTURE_VERSION,
        CHECKPOINT_VERSION,
        CONFIG_VERSION,
        SERIALIZATION_VERSION,
    )

try:
    from transformers import PretrainedConfig
except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
    raise ImportError("magi.hf.configuration_magi requires the optional transformers package") from exc


class MagiConfig(PretrainedConfig):
    model_type = "magi"

    def __init__(
        self,
        *,
        name: str = "MAGI",
        model_class: str = "dense_decoder",
        d_model: int = 512,
        n_layers: int = 8,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        d_head: int = 64,
        vocab_size: int = 32000,
        tied_embeddings: bool = True,
        rmsnorm_eps: float = 1.0e-6,
        rope_theta: float = 1000000.0,
        bias: bool = False,
        n_dense_layers: int | None = None,
        n_moe_layers: int | None = None,
        d_ff: int | None = None,
        d_ff_dense: int | None = None,
        d_ff_expert: int | None = None,
        n_routed_experts: int | None = None,
        n_shared_experts: int | None = None,
        moe_top_k: int | None = None,
        top_k: int | None = None,
        train_context: int | None = None,
        infer_context: int | None = None,
        architecture_version: str = ARCHITECTURE_VERSION,
        checkpoint_version: str = CHECKPOINT_VERSION,
        config_version: str = CONFIG_VERSION,
        serialization_version: str = SERIALIZATION_VERSION,
        moe_load_balance: str | None = None,
        router_z_loss_coeff: float | None = None,
        moe_bias_update_rate: float | None = None,
        moe_gate: str | None = None,
        **kwargs,
    ) -> None:
        # MoE routing width must not collide with HF sampling GenerationConfig.top_k.
        if moe_top_k is None and top_k is not None:
            moe_top_k = top_k
        kwargs.pop("top_k", None)
        moe_load_balance = kwargs.pop("moe_load_balance", moe_load_balance)
        router_z_loss_coeff = kwargs.pop("router_z_loss_coeff", router_z_loss_coeff)
        moe_bias_update_rate = kwargs.pop("moe_bias_update_rate", moe_bias_update_rate)
        moe_gate = kwargs.pop("moe_gate", moe_gate)
        kwargs.setdefault("architectures", ["MagiForCausalLM"])
        kwargs.setdefault("tie_word_embeddings", tied_embeddings)
        kwargs.setdefault("use_cache", True)
        kwargs.setdefault("return_dict", True)
        # transformers>=5 reads these aliases for cache/generation plumbing.
        if "num_hidden_layers" in kwargs:
            n_layers = int(kwargs.pop("num_hidden_layers"))
        if "hidden_size" in kwargs:
            d_model = int(kwargs.pop("hidden_size"))
        if "num_attention_heads" in kwargs:
            n_heads = int(kwargs.pop("num_attention_heads"))
        if "num_key_value_heads" in kwargs:
            n_kv_heads = int(kwargs.pop("num_key_value_heads"))
        if "head_dim" in kwargs:
            d_head = int(kwargs.pop("head_dim"))
        if "max_position_embeddings" in kwargs and infer_context is None and train_context is None:
            infer_context = int(kwargs.pop("max_position_embeddings"))
        else:
            kwargs.pop("max_position_embeddings", None)
        super().__init__(**kwargs)
        self.name = name
        self.model_class = model_class
        self.d_model = int(d_model)
        self.n_layers = int(n_layers)
        self.n_heads = int(n_heads)
        self.n_kv_heads = int(n_kv_heads)
        self.d_head = int(d_head)
        self.vocab_size = int(vocab_size)
        self.tied_embeddings = bool(tied_embeddings)
        self.tie_word_embeddings = bool(tied_embeddings)
        self.rmsnorm_eps = float(rmsnorm_eps)
        self.rope_theta = float(rope_theta)
        self.bias = bool(bias)
        self.n_dense_layers = _optional_int(n_dense_layers)
        self.n_moe_layers = _optional_int(n_moe_layers)
        self.d_ff = _optional_int(d_ff)
        self.d_ff_dense = _optional_int(d_ff_dense)
        self.d_ff_expert = _optional_int(d_ff_expert)
        self.n_routed_experts = _optional_int(n_routed_experts)
        self.n_shared_experts = _optional_int(n_shared_experts)
        self.moe_top_k = _optional_int(moe_top_k)
        self.train_context = _optional_int(train_context)
        self.infer_context = _optional_int(infer_context)
        self.moe_load_balance = None if moe_load_balance is None else str(moe_load_balance)
        self.router_z_loss_coeff = None if router_z_loss_coeff is None else float(router_z_loss_coeff)
        self.moe_bias_update_rate = None if moe_bias_update_rate is None else float(moe_bias_update_rate)
        self.moe_gate = None if moe_gate is None else str(moe_gate)
        self.architecture_version = str(architecture_version)
        self.checkpoint_version = str(checkpoint_version)
        self.config_version = str(config_version)
        self.serialization_version = str(serialization_version)
        # Never keep sampling top_k on architecture config; MoE width lives in moe_top_k.
        if "top_k" in self.__dict__:
            del self.__dict__["top_k"]
        # Prefer return_dict; drop deprecated use_return_dict if present on older dumps.
        if "use_return_dict" in self.__dict__:
            if "return_dict" not in self.__dict__:
                self.return_dict = bool(self.__dict__.pop("use_return_dict"))
            else:
                del self.__dict__["use_return_dict"]
        self.auto_map = {
            "AutoConfig": "configuration_magi.MagiConfig",
            "AutoModelForCausalLM": "modeling_magi.MagiForCausalLM",
        }
        self.validate()

    @property
    def num_hidden_layers(self) -> int:
        return int(self.n_layers)

    @num_hidden_layers.setter
    def num_hidden_layers(self, value: int) -> None:
        self.n_layers = int(value)

    @property
    def hidden_size(self) -> int:
        return int(self.d_model)

    @hidden_size.setter
    def hidden_size(self, value: int) -> None:
        self.d_model = int(value)

    @property
    def num_attention_heads(self) -> int:
        return int(self.n_heads)

    @num_attention_heads.setter
    def num_attention_heads(self, value: int) -> None:
        self.n_heads = int(value)

    @property
    def num_key_value_heads(self) -> int:
        return int(self.n_kv_heads)

    @num_key_value_heads.setter
    def num_key_value_heads(self, value: int) -> None:
        self.n_kv_heads = int(value)

    @property
    def head_dim(self) -> int:
        return int(self.d_head)

    @head_dim.setter
    def head_dim(self, value: int) -> None:
        self.d_head = int(value)

    @property
    def max_position_embeddings(self) -> int:
        return int(self.infer_context or self.train_context or 2048)

    @max_position_embeddings.setter
    def max_position_embeddings(self, value: int) -> None:
        self.infer_context = int(value)

    @classmethod
    def from_native_config(
        cls,
        cfg: ModelConfig,
        *,
        architecture_version: str = ARCHITECTURE_VERSION,
        checkpoint_version: str = CHECKPOINT_VERSION,
        config_version: str = CONFIG_VERSION,
        serialization_version: str = SERIALIZATION_VERSION,
    ) -> "MagiConfig":
        return cls(
            name=cfg.name,
            model_class=cfg.model_class,
            d_model=cfg.d_model,
            n_layers=cfg.n_layers,
            n_heads=cfg.n_heads,
            n_kv_heads=cfg.n_kv_heads,
            d_head=cfg.d_head,
            vocab_size=cfg.vocab_size,
            tied_embeddings=cfg.tied_embeddings,
            rmsnorm_eps=cfg.rmsnorm_eps,
            rope_theta=cfg.rope_theta,
            bias=cfg.bias,
            n_dense_layers=cfg.n_dense_layers,
            n_moe_layers=cfg.n_moe_layers,
            d_ff=cfg.d_ff,
            d_ff_dense=cfg.d_ff_dense,
            d_ff_expert=cfg.d_ff_expert,
            n_routed_experts=cfg.n_routed_experts,
            n_shared_experts=cfg.n_shared_experts,
            moe_top_k=cfg.top_k,
            train_context=cfg.train_context,
            infer_context=cfg.infer_context,
            architecture_version=architecture_version,
            checkpoint_version=checkpoint_version,
            config_version=config_version,
            serialization_version=serialization_version,
            moe_load_balance=cfg.moe_load_balance,
            router_z_loss_coeff=cfg.router_z_loss_coeff,
            moe_bias_update_rate=cfg.moe_bias_update_rate,
            moe_gate=cfg.moe_gate,
        )

    @classmethod
    def from_native_yaml(cls, path: str | Path) -> "MagiConfig":
        raw = load_simple_yaml(path)
        meta = raw.get("meta", {})
        return cls.from_native_config(
            load_model_config(path),
            config_version=str(meta.get("version", CONFIG_VERSION)),
        )

    def to_native_config(self) -> ModelConfig:
        cfg = ModelConfig(
            name=self.name,
            model_class=self.model_class,
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            d_head=self.d_head,
            vocab_size=self.vocab_size,
            tied_embeddings=self.tied_embeddings,
            rmsnorm_eps=self.rmsnorm_eps,
            rope_theta=self.rope_theta,
            bias=self.bias,
            n_dense_layers=self.n_dense_layers,
            n_moe_layers=self.n_moe_layers,
            d_ff=self.d_ff,
            d_ff_dense=self.d_ff_dense,
            d_ff_expert=self.d_ff_expert,
            n_routed_experts=self.n_routed_experts,
            n_shared_experts=self.n_shared_experts,
            top_k=self.moe_top_k,
            train_context=self.train_context,
            infer_context=self.infer_context,
            moe_load_balance=getattr(self, "moe_load_balance", None),
            router_z_loss_coeff=getattr(self, "router_z_loss_coeff", None),
            moe_bias_update_rate=getattr(self, "moe_bias_update_rate", None),
            moe_gate=getattr(self, "moe_gate", None),
        )
        cfg.validate()
        return cfg

    def to_native_yaml_text(self) -> str:
        cfg = self.to_native_config()
        fields = [
            ("meta", None),
            ("  name", cfg.name),
            ("  version", self.config_version),
            ("  program_parent", "MAGI-5.5GTBS"),
            ("  status", "hf_roundtrip"),
            ("  from_zero", "true"),
            ("  model_class", cfg.model_class),
            ("architecture", None),
            ("  d_model", cfg.d_model),
            ("  n_layers", cfg.n_layers),
            ("  n_heads", cfg.n_heads),
            ("  n_kv_heads", cfg.n_kv_heads),
            ("  d_head", cfg.d_head),
            ("  vocab_size", cfg.vocab_size),
            ("  tied_embeddings", str(cfg.tied_embeddings).lower()),
            ("  rmsnorm_eps", cfg.rmsnorm_eps),
            ("  rope_theta", cfg.rope_theta),
            ("  bias", str(cfg.bias).lower()),
        ]
        if cfg.is_moe:
            fields.extend(
                [
                    ("  type", "decoder_only_prenorm_sparse_moe"),
                    ("  n_dense_layers", cfg.n_dense_layers),
                    ("  n_moe_layers", cfg.n_moe_layers),
                    ("  d_ff_dense", cfg.d_ff_dense),
                    ("  d_ff_expert", cfg.d_ff_expert),
                    ("  n_routed_experts", cfg.n_routed_experts),
                    ("  n_shared_experts", cfg.n_shared_experts),
                    ("  top_k", cfg.top_k),
                ]
            )
        else:
            fields.extend([("  type", "decoder_only_prenorm_dense"), ("  d_ff", cfg.d_ff)])
        fields.extend([("  train_context", cfg.train_context), ("  infer_context", cfg.infer_context)])
        lines: list[str] = []
        for key, value in fields:
            if value is None:
                lines.append(f"{key}:")
            else:
                lines.append(f"{key}: {value}")
        return "\n".join(lines) + "\n"

    def save_native_yaml(self, path: str | Path) -> None:
        Path(path).write_text(self.to_native_yaml_text(), encoding="utf-8")

    def validate(self) -> None:
        self.to_native_config()
        for field_name in (
            "architecture_version",
            "checkpoint_version",
            "config_version",
            "serialization_version",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"MagiConfig.{field_name} must be an explicit non-empty string")


def _optional_int(value: int | str | None) -> int | None:
    return None if value is None else int(value)


def _register_auto_config() -> None:
    try:
        from transformers import AutoConfig
    except ImportError:
        return
    try:
        AutoConfig.register(MagiConfig.model_type, MagiConfig)
    except ValueError:
        registered = AutoConfig.for_model(MagiConfig.model_type)
        if registered is not MagiConfig:
            raise


_register_auto_config()

