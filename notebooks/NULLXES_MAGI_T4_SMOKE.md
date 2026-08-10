# NULLXES MAGI — T4 Colab Smoke

Tesla T4 (~15GB). Config: `configs/magi_t4_smoke_v0.1.yaml`.

## Cell 1 — setup

```python
!pip -q install torch transformers safetensors
!git clone https://github.com/MagistrTheOne/NULLXES-MAGI-5.5GTBS.git
%cd NULLXES-MAGI-5.5GTBS
```

## Cell 2 — validate config + tokenizer + GPU run

```python
!python scripts/validate_config.py --config configs/magi_t4_smoke_v0.1.yaml
!python scripts/param_count.py --config configs/magi_t4_smoke_v0.1.yaml
!python scripts/t4_smoke_run.py --device cuda --seq 256 --generate-tokens 16
```

## Cell 3 — quick HF path

```python
import torch
from magi.config import load_model_config
from magi.hf import MagiForCausalLM, native_config_to_hf
from magi.tokenizer import build_t4_smoke_tokenizer

cfg = load_model_config("configs/magi_t4_smoke_v0.1.yaml")
tok = build_t4_smoke_tokenizer(vocab_size=cfg.vocab_size)
model = MagiForCausalLM(native_config_to_hf(cfg)).to("cuda", dtype=torch.float16).eval()
ids = torch.tensor([tok.encode("MAGI T4", add_bos=True)], device="cuda")
print(model(input_ids=ids).logits.shape)
print("cuda_gb", torch.cuda.memory_allocated()/1024**3)
```
