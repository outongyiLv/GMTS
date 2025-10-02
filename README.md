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
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Upgrade basics
pip install -U pip wheel setuptools

# Install PyTorch (choose the right index URL for your CUDA/toolkit)
# Example for CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Core utilities (adjust as needed)
pip install transformers datasets accelerate tqdm numpy pyyaml
# Optional: visualization & notebooks
pip install matplotlib jupyter
# Optional: logging backends
# pip install wandb tensorboard


