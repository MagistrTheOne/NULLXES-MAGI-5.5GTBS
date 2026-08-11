# RunPod H200 — MAGI first intelligence init ($10)

**Image:** `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (PyTorch 2.8.0, cu128)  
**GPU:** 1× H200 SXM 141 GB  
**Repo:** https://github.com/MagistrTheOne/NULLXES-MAGI-5.5GTBS

---

## Model decision (фиксируем)

| Модель | На 1× H200 за $10 |
|--------|-------------------|
| MAGI-T4-SMOKE 116M | **основной train** — полный цикл |
| MAGI-CASUAL 13.789B | **load + forward + generate** (bf16). Full Adam train — впритык/OOM без ZeRO |
| MAGI-35B-MoE | только meta/param_count. Train нужен multi-GPU (tp/pp/ep) |
| MAGI-400B | **нет** (~813 GB bf16 weights) |

**Поднимаем:**
1. **Train / интеллект-init:** `MAGI-T4-SMOKE` + synthetic scale-up  
2. **Capacity proof:** `MAGI-CASUAL` dry load + forward  

Не путать: «H200 потянет любую» ≠ full train любой. Потянет **inference/load** mid-size; train — по памяти оптимизатора.

---

## Cell 0 — env

```bash
nvidia-smi
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory/1024**3)"
```

Ожидаемо: `2.8.0`, `H200`, ~141 GB.

---

## Cell 1 — deps (global, torch уже в образе)

```bash
pip -q install "transformers==4.57.1" safetensors huggingface_hub
# НЕ делать: pip install -U torch
```

HF token (когда дашь):

```bash
export HF_TOKEN="hf_..."   # или HUGGING_FACE_HUB_TOKEN
huggingface-cli login --token "$HF_TOKEN"
```

---

## Cell 2 — repo

```bash
cd /workspace
git clone https://github.com/MagistrTheOne/NULLXES-MAGI-5.5GTBS.git || true
cd NULLXES-MAGI-5.5GTBS
git fetch origin && git checkout main && git pull --ff-only origin main
export PYTHONPATH=/workspace/NULLXES-MAGI-5.5GTBS:$PYTHONPATH
```

---

## Cell 3 — validate stack

```bash
python -c "import torch, transformers; from magi.hf import MagiForCausalLM, HF_AVAILABLE; print(torch.__version__, transformers.__version__, HF_AVAILABLE, MagiForCausalLM)"
python scripts/validate_config.py --config configs/magi_t4_smoke_v0.1.yaml
python scripts/param_count.py --config configs/magi_t4_smoke_v0.1.yaml
python scripts/param_count.py --config configs/magi_casual_v0.1.yaml
python scripts/validate_all_models.py
```

---

## Cell 4 — synthetic corpus (scale)

```bash
python scripts/build_synthetic_dataset.py --docs 20000 --seed 42 --seq 512 \
  --output artifacts/synthetic/magi_synth_h200_v0.1
```

---

## Cell 5 — TRAIN (основной интеллект-init)

fp32 master + bf16/fp16 autocast. На H200 можно поднять seq.

```bash
python scripts/t4_train_smoke.py --device cuda --steps 500 --seq 512 \
  --corpus artifacts/synthetic/magi_synth_h200_v0.1/records.jsonl \
  --checkpoint-dir artifacts/h200_t4_train
```

Если стабильно и есть время:

```bash
python scripts/t4_train_smoke.py --device cuda --steps 2000 --seq 1024 \
  --corpus artifacts/synthetic/magi_synth_h200_v0.1/records.jsonl \
  --checkpoint-dir artifacts/h200_t4_train
```

Успех: `loss_improved=true`, `status=OK`, ckpt в `artifacts/h200_t4_train/`.

---

## Cell 6 — CASUAL capacity proof (не full Adam)

```bash
python - <<'PY'
import torch
from magi.config import load_model_config
from magi.model import MAGITransformer

cfg = load_model_config("configs/magi_casual_v0.1.yaml")
print(cfg.name, "params", "d_model", cfg.d_model, "layers", cfg.n_layers, "vocab", cfg.vocab_size)
# bf16 weights only — inference-style footprint
model = MAGITransformer.from_config(cfg).to(device="cuda", dtype=torch.bfloat16).eval()
n = sum(p.numel() for p in model.parameters())
print("numel", n, "bf16_gb", n * 2 / 1024**3)
ids = torch.randint(0, min(1000, cfg.vocab_size), (1, 64), device="cuda")
with torch.no_grad():
    out = model(ids, return_dict=True)
print("logits", tuple(out.logits.shape))
print("alloc_gb", torch.cuda.memory_allocated()/1024**3)
print("status=CASUAL_FORWARD_OK")
PY
```

---

## Cell 7 — optional HF wrap smoke (T4 cfg)

```bash
python scripts/t4_smoke_run.py --device cuda --seq 256 --generate-tokens 32
```

---

## Time box ($10)

Смотри актуальный $/hr H200 в RunPod. Ориентир:
- 0–15 мин: clone + deps + validate  
- 15–40 мин: synthetic 20k  
- остаток: train 500–2000 steps  
- последние 10–15 мин: CASUAL forward proof + скачать ckpt

```bash
# скачать ckpt на локаль / объектку до kill pod
ls -lh artifacts/h200_t4_train/
```

---

## Stop rules

- NaN loss → stop  
- OOM на CASUAL train → не форсировать; capacity proof достаточно  
- не ставить `pip install -U torch`  
- не запускать 35B/400B train на 1× H200
