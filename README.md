# GMTS-Framework1

> Lightweight, single-GPU training & evaluation pipeline for GMTS experiments — **no VLLM/SGLang required**, minimal env constraints, ideal for resource-constrained setups.

---

## ✨ Highlights

- **No VLLM/SGLang required** pure PyTorch and transformer pipeline for training and evaluation.
- **Minimal dependency constraints** strict requirements on framework, packages, or CUDA versions.
- **Single-GPU operation** Designed without data/model parallelism, making it practical under limited resources.

> ⚠️ As the framework omits parallelization, **its throughput is lower than verl**. It is best suited for prototype exploration, ablation studies, and quick feasibility checks under resource constraints.
---

## 🚀 Start

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

---

##### 🔧 `model` section

| Key | Example | Type | Description |
|:---:|:---:|:---:|:---:|
| `pretrained_model_path` | `./GMTS/model/Qwen2.5-Math-1.5B` | str | Path or HF identifier of the base model to train for RLVR. |
| `device` | `cuda:0` | str | Primary compute device for the **policy/current** model. |
| `dtype` | `bfloat16` | str | Compute dtype for model weights/forward. |
| `old_model_device` | `cuda:0` | str | Device for the old model. |
| `ref_model_device` | `cuda:0` | str | Device for the **reference** model used in KL for GRPO. |

---

##### 📚 `data` section

| Key | Example | Type | Description |
|:---:|:---:|:---:|:---:|
| `data_name` | `math` | str | Dataset used for routing or logging. |
| `training_path` | `./GMTS/GMTS-Framework1/test_data/math-500` | str | Directory or file path to training data. |
| `testing_path` | `./GMTS/GMTS-Framework1/test_data/math-500` | str | Directory or file path to testing data. |
| `test_size` | `500` | int | Number of test examples to load/evaluate. |

---

##### 🧪 `testing` section

| Key | Example | Type | Description |
|:---:|:---:|:---:|:---:|
| `batch_size` | `125` | int | Batch size used during **evaluation**. |

---

##### 🏋️ `training` section

| Key | Example | Type | Description |
|:---:|:---:|:---:|:---:|
| `method` | `dapo` | str | Training method. |
| `random_seed` | `0` | int | Global seed for training. |
| `max_prompt_len` | `1024` | int | Max input prompt tokens. |
| `max_gen_len` | `2048` | int | Max generated output tokens per sample. |
| `batch_size` | `1024` | int | Global training dataset number **Group number** is `batch_size`//`num_questions_per_batch`. |
| `num_questions_per_batch` | `64` | int | Number of questions per step. |
| `rollout_mini_batch` | `16` | int | Questions per rollout chunk. |
| `mini_batch_size` | `64` | int | Questions per **update** chunk. |
| `micro_batch_size` | `4` | int | Per-GPU micro-batch size for gradient accumulation. |
| `kl_beta` | `0.001` | float | KL coefficient. |
| `learning_rate` | `3.0e-5` | float | Optimizer learning rate. |
| `ckpt_dir` | `./GMTS/GMTS-Framework1/result/ckpt_dir/Qwen2.5-math-1.5b-dapo` | str | Directory to save checkpoints. |
| `log_dir` | `./GMTS/GMTS-Framework1/result/log_dir/Qwen2.5-math-1.5b-dapo` | str | Directory to write training logs/metrics. |
| `save_logp_gradient_path` | `./GMTS/.../gradient_dir/Qwen2.5-math-1.5b-dapo/` | str | Directory for saving logp-gradient magnitude. |
| `save_true_gradient_path` | `./GMTS/.../gradient_dir/Qwen2.5-math-1.5b-dapo/` | str | Directory for saving true-gradient magnitude. |
| `save_entrpy_path` | `./GMTS/.../gradient_dir/Qwen2.5-math-1.5b-dapo/` | str | Directory for saving entropy. |
| `save_mask_path` | `./GMTS/.../gradient_dir/Qwen2.5-math-1.5b-dapo/` | str | Directory for saving each token's mask. |
| `save_adv_pth` | `./GMTS/.../gradient_dir/Qwen2.5-math-1.5b-dapo/` | str | Directory for saving advantage. |
| `save_prefix_pth` | `./GMTS/.../gradient_dir/Qwen2.5-math-1.5b-dapo/` | str | Directory for saving prefix information. |
| `save_ids_path` | `./GMTS/.../gradient_dir/Qwen2.5-math-1.5b-dapo/` | str | Directory for saving token IDs. |
| `ckpt_save_interval` | `200` | int | Save a checkpoint every N training steps. |
| `gradient_interval` | `100` | int | Write gradient and gradient information every N steps. |
| `eval_interval` | `10` | int | Run evaluation every N steps. |
| `max_grad_norm` | `1.0` | float | Gradient clipping threshold (L2 norm). |
| `weight_decay` | `0.0` | float | L2 weight decay coefficient. |
| `betas` | `[0.9, 0.999]` | list[float] | AdamW momentum coefficients. |
| `memory_efficient_adamw` | `false` | bool | Use a memory-optimized AdamW if supported. |
| `clip_ratio_low` | `0.20` | float | Lower ratio bound. |
| `clip_ratio_high` | `0.28` | float | Upper ratio bound. |
| `use_entropy` | `false` | bool | Whether to use **ETS**/**GMTS**. |
| `selected_percent` | `0.0` | float | Selected ratio for **ETS**/**GMTS**. |
| `use_TES_method` | `false` | bool  | Whether to use **ETS**. |
| `use_GMTS_method` | `false` | bool | Whether to use **GMTS**. |
| `read_model_path` | `""` | str | Load weights from a specific checkpoint before training. |
| `doing_inverse` | `false` | bool | Whether doing bottom selection. |
| `use_filter` | `false` | bool | Apply filtering on samples before training. |

---

### 3) Evaluation
> Before running `test.py`
```bash
python3 test.py
```
