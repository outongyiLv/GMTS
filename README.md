# GMTS-Framework1

> Lightweight, single-GPU training & evaluation pipeline for GMTS experiments — **no VLLM/SGLang required**, minimal env constraints, ideal for resource-constrained setups.

---

## ✨ Highlights

- **No VLLM/SGLang required** pure PyTorch and transformer pipeline for training and evaluation.
- **Minimal dependency constraints** strict requirements on framework, packages, or CUDA versions.
- **Single-GPU operation** Designed without data/model parallelism, making it practical under limited resources.

> ⚠️ As the framework omits parallelization, **its throughput is lower than verl**. It is best suited for prototype exploration, ablation studies, and quick feasibility checks under resource constraints.
---

## 🚀 快速开始

### 1) 环境准备（示例）
```bash
# 建议 Python 3.9+
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -U pip wheel
# 下面库仅为示例，按需增减；可与本机 CUDA 匹配安装 torch/torchvision
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate tqdm numpy pyyaml
pip install matplotlib jupyter


