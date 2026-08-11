# RunPod H200 — MAGI sparse MoE architecture bring-up

**Image:** `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` (PyTorch 2.8.0)  
**GPU:** 1× H200 SXM 141 GB  
**Repo:** https://github.com/MagistrTheOne/NULLXES-MAGI-5.5GTBS

---

## ENGINEERING SCALING RULE

Reducing parameter count is allowed.  
Changing the computational architecture is **not**.  
Dense models are baselines only — never the canonical MoE research path.

---

## Model (канон)

**Train target: MAGI-7B-MoE — sparse MoE, topology scaled from MAGI-35B-MoE.**

| Модель | На 1× H200 |
|--------|------------|
| **MAGI-7B-MoE (~6.46B total / ~0.97B active)** | **FULL AdamW train — канон** |
| MAGI-7B-DENSE-BASELINE | только comparison baseline |
| MAGI-CASUAL 13.8B | dense capacity proof |
| MAGI-35B / 400B | не на 1 GPU full Adam |
| MAGI-T4-SMOKE | kernel smoke |

Сохранено из 35B: `moe_decoder`, GQA, dense prefix + MoE stack, 64 routed / 1 shared, `top_k=4`, `d_ff_expert=512`, `sigmoid_normalize_topk`, те же `MoERouter`/`MoELayer`, telemetry `router_entropy` / `dead_experts` / `imbalance`.

Checkpoint: **`model.safetensors`**.

---

## SSH

```bash
ssh root@103.196.86.21 -p 43139 -i ~/.ssh/id_ed25519
# or: ssh runpod-magi-h200
```

---

## Commands

```bash
cd /workspace/NULLXES-MAGI-5.5GTBS
git pull --ff-only origin main
export PYTHONPATH=$PWD:$PYTHONPATH
pip -q install -r requirements-h200.txt

python scripts/param_count.py --config configs/magi_7b_moe_v0.1.yaml
python scripts/validate_config.py --config configs/magi_7b_moe_v0.1.yaml

python scripts/build_synthetic_dataset.py --docs 20000 --seed 42 --seq 512 \
  --output artifacts/synthetic/magi_synth_h200_v0.1

# MAIN — MAGI-7B-MoE
python scripts/h200_train.py --device cuda --steps 2000 --seq 2048 \
  --corpus artifacts/synthetic/magi_synth_h200_v0.1/records.jsonl \
  --checkpoint-dir artifacts/h200_7b_moe_train \
  --checkpoint-every 100
```

Успех: `model=MAGI-7B-MoE`, `router_entropy=<число>`, `dead_experts=...`, ckpt в `artifacts/h200_7b_moe_train/model.safetensors`.

Если OOM на seq 2048 → `--seq 1024`.

---

## Stop rules

- `router_entropy=n/a` на этом профиле = **баг** (должен быть MoE)  
- NaN → stop  
- не `-U torch`  
- не запускать dense 7B как H200 research path  
- скачать: `ls -lh artifacts/h200_7b_moe_train/model.safetensors artifacts/h200_7b_moe_train/step-*/`
