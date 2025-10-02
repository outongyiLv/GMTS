# GMTS-Framework1

> Lightweight, single-GPU training & evaluation pipeline for GMTS experiments — **no VLLM/SGLang required**, minimal env constraints, ideal for resource-constrained setups.

---

## ✨ Highlights

- **No VLLM/SGLang required** pure PyTorch and transformer pipeline for training and evaluation.
- **Minimal dependency constraints** strict requirements on framework, packages, or CUDA versions.
- **Single-GPU operation** Designed without data/model parallelism, making it practical under limited resources.

> ⚠️ As the framework omits parallelization, **its throughput is lower than verl**. It is best suited for prototype exploration, ablation studies, and quick feasibility checks under resource constraints.
---

## 🚀 Quick Start

### 1) Environment Setup

> There are **no strict constraints** on CUDA or package versions; choose a PyTorch build compatible with your system.

```bash
git clone https://github.com/outongyiLv/GMTS.git
conda create -n GMTS python=3.10 -y
conda activate GMTS
cd GMTS/GMTS-Framework1
pip install -r requirements.txt
```

### 2) Running GMTS on Qwen2.5-math-1.5B
```bash
cd GMTS/GMTS-Framework1
python3 train.py --config ./example/Qwen2.5-math-1.5b-grpo-GMTS.yaml
```
#### ⚙️ Common Arguments & Meanings
> You can customize your own training; the specific parameters in **.yaml** are as follows:


##### 🔧 `model` section

| Key | Example | Type | Description |
|---|---|---:|---|
| `pretrained_model_path` | `./GMTS/model/Qwen2.5-Math-1.5B` | str | Path or HF identifier of the base model to train for RLVR. |
| `device` | `cuda:0` | str | Primary compute device for the **policy/current** model. |
| `dtype` | `bfloat16` | str | Compute dtype for model weights/forward. |
| `old_model_device` | `cuda:0` | str | Device for the old model (can be the same as `device` if memory allows). |
| `ref_model_device` | `cuda:0` | str | Device for the **reference** model used in KL for GRPO (can be the same as `device` if memory allows). |

