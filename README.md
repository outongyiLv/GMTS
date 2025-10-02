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


| 参数名 | 类型(默认) | 说明 |
|---|---|---|
| `--model_name_or_path` | str | 基座模型（HF 模型名或本地路径），如 `Qwen/Qwen2.5-Math-1.5B` |
| `--train_file` | str | 训练集路径（json/jsonl/自定义） |
| `--eval_file` | str/可选 | 验证集路径 |
| `--output_dir` | str | 输出目录（ckpt/日志/图表） |
| `--epochs` | int (1) | 训练轮数 |
| `--lr` | float (1e-5) | 学习率 |
| `--batch_size` | int (1) | 全局 batch（单卡） |
| `--grad_accum_steps` | int (8) | 梯度累积步数（折中显存/吞吐） |
| `--max_len` | int (2048) | 最大序列长度 |
| `--dtype` | str (`auto`) | 精度：`fp16`/`bf16`/`auto` |
| `--seed` | int (42) | 随机种子 |
| `--log_interval` | int (50) | 训练日志打印间隔（steps） |
| `--eval_interval` | int (500) | 评估频率（steps） |
| `--save_interval` | int (1000) | 保存频率（steps） |
| `--wandb` | flag | 启用 Weights & Biases 记录 |
| `--exp_name` | str | 实验名（便于区分 run） |
