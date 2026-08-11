# RunPod H200 — MAGI intelligence bring-up ($10)

**Image:** `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (PyTorch 2.8.0)  
**GPU:** 1× H200 SXM 141 GB  
**Repo:** https://github.com/MagistrTheOne/NULLXES-MAGI-5.5GTBS

---

## Model (фиксируем)

**Train target: MAGI-7B — 6.413B dense.**

| Модель | На 1× H200 |
|--------|------------|
| **MAGI-7B (6.413B)** | **FULL AdamW train** — основной run |
| MAGI-CASUAL 13.789B | forward/load ok; full Adam почти в потолок 141GB |
| MAGI-35B / 400B | train на 1 GPU — нет (нужен cluster / ZeRO) |
| MAGI-T4-SMOKE 116M | только kernel smoke, **не** H200 workhorse |

Почему не «70–100B»: bf16 веса 70B ≈ 140GB — это **inference fill**.  
Full Adam from-scratch ≈ **8 байт/параметр** на m+v → 70B train ≈ **560GB+** только оптимизатор. На 1×H200 без ZeRO это невозможно. 7B — честный full-train на карте.

VRAM 7B (оценка): bf16 weights ~13GB + Adam fp32 ~51GB + activations seq2k ≈ **<100GB**.

---

## Commands

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory/1024**3,1))"

pip -q install "transformers==4.57.1" safetensors huggingface_hub
# НЕ: pip install -U torch

cd /workspace
git clone https://github.com/MagistrTheOne/NULLXES-MAGI-5.5GTBS.git || true
cd NULLXES-MAGI-5.5GTBS
git pull --ff-only origin main
export PYTHONPATH=$PWD:$PYTHONPATH

# HF (когда будет токен):
# export HF_TOKEN=hf_...
# huggingface-cli login --token "$HF_TOKEN"

python scripts/param_count.py --config configs/magi_7b_v0.1.yaml
python scripts/validate_config.py --config configs/magi_7b_v0.1.yaml

python scripts/build_synthetic_dataset.py --docs 20000 --seed 42 --seq 512 \
  --output artifacts/synthetic/magi_synth_h200_v0.1

# MAIN TRAIN — MAGI-7B
python scripts/h200_train.py --device cuda --steps 2000 --seq 2048 \
  --corpus artifacts/synthetic/magi_synth_h200_v0.1/records.jsonl \
  --checkpoint-dir artifacts/h200_7b_train

# If VRAM tight:
# python scripts/h200_train.py --device cuda --steps 2000 --seq 1024 --corpus ...
```

Успех: `model=MAGI-7B`, `loss_improved=true`, ckpt в `artifacts/h200_7b_train/`.

---

## Optional CASUAL capacity proof (не train)

```bash
python - <<'PY'
import torch
from magi.config import load_model_config
from magi.model import MAGITransformer
cfg = load_model_config("configs/magi_casual_v0.1.yaml")
model = MAGITransformer.from_config(cfg).to(device="cuda", dtype=torch.bfloat16).eval()
n = sum(p.numel() for p in model.parameters())
print("CASUAL", n, "bf16_gb", n*2/1024**3)
ids = torch.randint(0, 1000, (1, 64), device="cuda")
with torch.no_grad():
    print("logits", model(ids, return_dict=True).logits.shape)
print("alloc_gb", torch.cuda.memory_allocated()/1024**3)
PY
```

---

## Stop rules

- NaN → stop  
- OOM на seq 2048 → `--seq 1024` или `512`  
- не ставить `-U torch`  
- не запускать 35B/400B train на 1×H200  
- скачать ckpt до kill pod: `ls -lh artifacts/h200_7b_train/`
