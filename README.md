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


