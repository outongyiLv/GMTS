set -x
export VLLM_ATTENTION_BACKEND = XFORMERS                       # no need to change.
export TOKENIZERS_PARALLELISM = 'true'                         # no need to change.

train_files="./data/process_data/math/train.parquet"           # training files for math.
test_files="./data/process_data/math/test.parquet"             # testing files for math.

adv_estimator=grpo                                             
train_batch_size=512                                           
max_prompt_length=1024                                         
max_response_length=4096                                       
ppo_mini_batch_size=32                                         
rollout_n=16                                                   # group-size大小

model_path="./model/Qwen3-8B"                                  # model
model_lr=1e-6                                                  # lr
use_kl_loss=False                                              # GRPO needs
kl_loss_coef=0.0                                               # GRPO's kl-coef'
offload=False                                                  # cpu or gpu

loss_agg_mode="token-mean"                                     
enable_filter_groups=False                                     
filter_groups_metric=acc                                       
max_num_gen_batches=10                                         

tensor_model_parallel_size=4                                   
seed=0                                                         
gpu_memory_utilization=0.6                                     

inf_temperature=1.0                                            
project_name='2025-09-07'
experiment_name="2025-09-07-DAPO-verl-Qwen3-8B-Math-GMTS"

GPU_number=8                                                   
Node_number=1                                                  
save_freq=5                                                    
test_freq=1                                                    

val_before_train=True                                          
total_epochs=1                                                 

use_dynamic_bsz=True                                           
use_kl_in_reward=False                                         
filter_overlong_prompts=False                                  

doing_entropy_clipping_type="GMTS"                             
doing_entropy_clipping_percent=0.8                             

clip_ratio_low=0.2
clip_ratio_high=0.28

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m recipe.dapo.src.main_dapo \
                                                algorithm.adv_estimator="$adv_estimator" \
                                                algorithm.use_kl_in_reward=$use_kl_in_reward \
                                                algorithm.doing_entropy_clipping_type=$doing_entropy_clipping_type\
                                                algorithm.doing_entropy_clipping_percent=$doing_entropy_clipping_percent\
                                                algorithm.filter_groups.enable=${enable_filter_groups} \
                                                algorithm.filter_groups.metric=${filter_groups_metric} \
                                                algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
                                                data.train_files="$train_files" \
                                                data.val_files="$test_files" \
                                                data.train_batch_size=$train_batch_size \
                                                data.max_prompt_length=$max_prompt_length \
                                                data.max_response_length=$max_response_length \
                                                data.filter_overlong_prompts=$filter_overlong_prompts \
                                                data.truncation='left' \
                                                reward_model.reward_manager=dapo \
                                                actor_rollout_ref.model.path=$model_path  \
                                                actor_rollout_ref.actor.optim.lr=$model_lr \
                                                actor_rollout_ref.model.use_remove_padding=True \
                                                actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
                                                actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
                                                actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
                                                actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
                                                actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
                                                actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
                                                actor_rollout_ref.actor.kl_loss_type=low_var_kl \
                                                actor_rollout_ref.actor.entropy_coeff=0 \
                                                actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
                                                actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
                                                actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
                                                actor_rollout_ref.model.enable_gradient_checkpointing=True \
                                                actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
                                                actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
                                                actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
                                                actor_rollout_ref.rollout.name=vllm \
                                                actor_rollout_ref.rollout.engine_seed=$seed \
                                                actor_rollout_ref.rollout.gpu_memory_utilization=$gpu_memory_utilization \
                                                actor_rollout_ref.rollout.n=$rollout_n \
                                                actor_rollout_ref.rollout.val_kwargs.do_sample=False \
                                                actor_rollout_ref.rollout.val_kwargs.n=1 \
                                                actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
                                                actor_rollout_ref.rollout.temperature=$inf_temperature \
                                                trainer.logger=['console','swanlab'] \
                                                trainer.project_name="$project_name" \
                                                trainer.experiment_name="$experiment_name" \
                                                trainer.n_gpus_per_node=$GPU_number \
                                                trainer.nnodes=$Node_number \
                                                trainer.save_freq=$save_freq \
                                                trainer.test_freq=$test_freq \
                                                trainer.val_before_train=$val_before_train \
                                                trainer.total_epochs=$total_epochs $@