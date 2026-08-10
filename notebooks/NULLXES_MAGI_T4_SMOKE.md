# NULLXES MAGI — T4 Colab Smoke

Tesla T4 (~15GB). Config: `configs/magi_t4_smoke_v0.1.yaml`.

## Cell 1 — setup (pin transformers 4.57.1)

Do **not** `pip install -U torch` on Colab. transformers 5.x needs torch>=2.5 and breaks GenerationMixin when the torch backend is disabled.

```python
!pip -q install "transformers==4.57.1" safetensors
import os
os.kill(os.getpid(), 9)  # hard-restart runtime so the new transformers is loaded
```

## Cell 2 — pull repo + run

Re-run after the runtime restart from Cell 1.

```python
!git clone https://github.com/MagistrTheOne/NULLXES-MAGI-5.5GTBS.git || true
%cd /content/NULLXES-MAGI-5.5GTBS
!git fetch origin && git checkout main && git pull --ff-only origin main
!python -c "import torch, transformers; from magi.hf import MagiForCausalLM, HF_AVAILABLE, HF_IMPORT_ERROR, format_hf_import_error; print('torch', torch.__version__); print('tf', transformers.__version__); print('hf', HF_AVAILABLE, MagiForCausalLM); print('err', format_hf_import_error() if HF_IMPORT_ERROR else None)"
!python scripts/validate_config.py --config configs/magi_t4_smoke_v0.1.yaml
!python scripts/param_count.py --config configs/magi_t4_smoke_v0.1.yaml
!python scripts/validate_all_models.py
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
