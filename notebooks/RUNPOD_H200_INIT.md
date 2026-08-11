# RunPod H200 — MAGI intelligence bring-up ($10)

**Image:** `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (PyTorch 2.8.0)  
**GPU:** 1× H200 SXM 141 GB  
**Repo:** https://github.com/MagistrTheOne/NULLXES-MAGI-5.5GTBS

---

## Model

**Train target: MAGI-7B — 6.413B dense.**

| Модель | На 1× H200 |
|--------|------------|
| **MAGI-7B (6.413B)** | **FULL AdamW train** |
| MAGI-CASUAL 13.789B | forward/load ok; full Adam почти потолок |
| MAGI-35B / 400B | не на 1 GPU |
| MAGI-T4-SMOKE 116M | только kernel smoke |

Checkpoint: **`model.safetensors`** (канон). `optimizer.pt` — опционально (`--save-optimizer`, ~50GB на 7B).

---

## Commands

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory/1024**3,1))"

cd /workspace
git clone https://github.com/MagistrTheOne/NULLXES-MAGI-5.5GTBS.git || true
cd NULLXES-MAGI-5.5GTBS
git pull --ff-only origin main
export PYTHONPATH=$PWD:$PYTHONPATH

pip -q install -r requirements-h200.txt
# Optional flash-attn (иначе PyTorch SDPA flash/mem-efficient):
# pip install flash-attn --no-build-isolation

python scripts/param_count.py --config configs/magi_7b_v0.1.yaml

python scripts/build_synthetic_dataset.py --docs 20000 --seed 42 --seq 512 \
  --output artifacts/synthetic/magi_synth_h200_v0.1

# MAIN — mid ckpt every 100 steps → artifacts/h200_7b_train/step-XXXXXX/model.safetensors
# Ctrl+C → пишет final/latest model.safetensors
python scripts/h200_train.py --device cuda --steps 2000 --seq 2048 \
  --corpus artifacts/synthetic/magi_synth_h200_v0.1/records.jsonl \
  --checkpoint-dir artifacts/h200_7b_train \
  --checkpoint-every 100

# Resume:
# python scripts/h200_train.py --device cuda --steps 2000 --seq 2048 \
#   --corpus artifacts/synthetic/magi_synth_h200_v0.1/records.jsonl \
#   --checkpoint-dir artifacts/h200_7b_train \
#   --resume artifacts/h200_7b_train/step-000400
```

Успех: `model=MAGI-7B`, `loss_improved=true`, веса в `artifacts/h200_7b_train/model.safetensors`.

Synth corpus — bring-up only; loss →0 = memorization, не intelligence. Для реального run нужен внешний corpus.

---

## Optional CASUAL capacity proof

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
- OOM на seq 2048 → `--seq 1024`  
- не `-U torch`  
- не 35B/400B train на 1×H200  
- скачать до kill: `ls -lh artifacts/h200_7b_train/model.safetensors artifacts/h200_7b_train/step-*/`
