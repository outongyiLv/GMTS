model_dirs=("/rgzn/home/zyw/loty/loty-main/checkpoints/models/DAPO-Qwen3-8b-QWA/33")


device=0
template="our"
temperatures="1.0"
N_SAMPLING=16
top_p=1.0

use_chat=1
use_system_prompt=1

echo "Model directories:"

for model_dir in "${model_dirs[@]}"; do
    echo "${model_dir}"
done

seeds="0"
for model_dir in "${model_dirs[@]}"; do
    CUDA_VISIBLE_DEVICES=${device} python eval_baseline.py \
        --model_name ${model_dir} \
        --template ${template} \
        --seeds ${seeds} \
        --temperatures ${temperatures} \
        --top_p        ${top_p}\
        --n_samples    ${N_SAMPLING} \
        --save True
done