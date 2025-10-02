# GMTS-Framework1

> Lightweight, single-GPU training & evaluation pipeline for GMTS experiments — **no VLLM/SGLang required**, minimal env constraints, ideal for quick iterations and resource-constrained setups.

---

## ✨ Highlights

- **无需 VLLM / SGLang**：纯 PyTorch 训练与测试流程。
- **对框架、包与 CUDA 版本无强依赖**：更易在不同机器上跑通（建议使用较新的 PyTorch + CUDA 组合以获得更好性能）。
- **单卡即可运行**：不包含数据/模型并行；在资源有限时尤为友好。
- **可视化分析**：提供 `logp–entropy`、`logp–gradient-magnitude`、`log-true-gradient-magnitude` 等关系图的分析工具。
- **上手即用示例**：`example/` 下含简单可复现的 run-example。

> ⚠️ 由于不使用并行化策略，本框架的**吞吐与效率低于 VERL**；适合**原型探索与对比实验**，或在资源紧张时快速验证想法。

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


