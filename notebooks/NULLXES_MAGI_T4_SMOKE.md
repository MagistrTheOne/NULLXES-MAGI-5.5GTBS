# NULLXES MAGI — T4 Colab Smoke

Tesla T4 (~15GB). Config: `configs/magi_t4_smoke_v0.1.yaml`.

## Preferred: Factory reset runtime

Runtime → Disconnect and delete runtime → Connect (GPU T4).  
Do **not** `pip install -U torch`.

## Cell 1 — deps + align torchvision + restart

If the runtime was previously polluted by `pip install -U torch`, torchvision breaks with:
`RuntimeError: operator torchvision::nms does not exist`.

```python
# Keep current torch 2.13+cu130; install matching torchvision. Do not -U torch again.
!pip -q install "transformers==4.57.1" safetensors
!pip -q install --force-reinstall "torchvision==0.28.0" --index-url https://download.pytorch.org/whl/cu130
import os
os.kill(os.getpid(), 9)
```

## Cell 2 — pull + smoke

```python
!git clone https://github.com/MagistrTheOne/NULLXES-MAGI-5.5GTBS.git || true
%cd /content/NULLXES-MAGI-5.5GTBS
!git fetch origin && git checkout main && git pull --ff-only origin main
!python -c "import torch, torchvision, transformers; from magi.hf import MagiForCausalLM, HF_AVAILABLE, HF_IMPORT_ERROR, format_hf_import_error; print('torch', torch.__version__); print('tv', torchvision.__version__); print('tf', transformers.__version__); print('hf', HF_AVAILABLE, MagiForCausalLM); print('err', format_hf_import_error() if HF_IMPORT_ERROR else None)"
!python scripts/validate_config.py --config configs/magi_t4_smoke_v0.1.yaml
!python scripts/param_count.py --config configs/magi_t4_smoke_v0.1.yaml
!python scripts/validate_all_models.py
!python scripts/t4_smoke_run.py --device cuda --seq 256 --generate-tokens 16
```

Expected check line: `hf True <class 'magi.hf.modeling_magi.MagiForCausalLM'>`.

## Cell 3 — training smoke (native AdamW, 20 steps)

```python
!python scripts/t4_train_smoke.py --device cuda --steps 20 --seq 128
```

Expected: `loss_improved=true`, `status=OK`, checkpoint under `artifacts/t4_train_smoke/`.

## Cell 4 — quick HF path

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
