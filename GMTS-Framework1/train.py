import html
import time
import numpy as np
import torch
import yaml

from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from grpo_tools.tokenizer import Tokenizer               
from grpo_tools.optimizer import MemoryEfficientAdamW    
from grpo_tools.qwen2_model import Transformer           
from grpo_tools.task_type.math_task_QW import mathTasksDataset, math_reward_function
from grpo_tools.grpo import rollout, update_policy, gradient


def evaluate(model, tokenizer, device, dtype, config):
    
    evalute_dataset_name   = ["math"]
    evalute_dataset_method = ["pass1"]
    evalute_dataset_path   = ["/home/zyw/GMTS/GMTS-Framework1/test_data/math-500"]
    evaluate_acc_result    = [0 for i in range(len(evalute_dataset_path))]
    evaluate_length_result = [0 for i in range(len(evalute_dataset_path))]

    for i in range(len(evalute_dataset_name)):
        method = evalute_dataset_method[i] # get the method

        test_dataset = mathTasksDataset(
                train_data_path=evalute_dataset_path[i],
                test_data_path=evalute_dataset_path[i],
                tokenizer=tokenizer,
                split="test",
                data_name = evalute_dataset_name[i]
            )
        
        generator = torch.Generator(device=device)
        
        dataloader = DataLoader(
            test_dataset,
            shuffle=False,
            collate_fn=mathTasksDataset.collate_fn,
            generator=generator,
            batch_size=config["testing"]["batch_size"],
            drop_last=False,
        )

        
        if(method=="avg16"):

            for num_number in range(16): 
                success = []
                token_length = []
                
                for batch in dataloader:
                    episodes = rollout(
                        model=model,
                        tokenizer=tokenizer,
                        batch=batch,
                        max_gen_len=config["training"]["max_gen_len"],
                        num_answer_per_question=1,
                        reward_function=math_reward_function,
                        device=device,
                        dtype=dtype,
                        data_name= evalute_dataset_name[i]
                    )
                    
                    success.extend([episode.reward_info["answer_reward"] for episode in episodes])
                    token_length.extend([len(episode.generated_token_ids) for episode in episodes])
                    
                evaluate_acc_result[i] += np.mean(success) 
                evaluate_length_result[i] += np.mean(token_length)
        
        evaluate_acc_result[i] = evaluate_acc_result[i]/16
        evaluate_length_result[i] = evaluate_length_result[i]/16

        
        if(method=="pass1"):
            success = []
            token_length = []
            
            for batch in dataloader:
                episodes = rollout(
                    model=model,
                    tokenizer=tokenizer,
                    batch=batch,
                    max_gen_len=config["training"]["max_gen_len"],
                    num_answer_per_question=1,
                    reward_function=math_reward_function,
                    device=device,
                    dtype=dtype,
                    data_name= evalute_dataset_name[i]
                )
                
                success.extend([episode.reward_info["answer_reward"]  for episode in episodes])
                token_length.extend([len(episode.generated_token_ids) for episode in episodes])
                
            evaluate_acc_result[i] += np.mean(success)
            evaluate_length_result[i] += np.mean(token_length)

    return evaluate_acc_result, evaluate_length_result


def main(config_path: str):

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # training-setting:
    rollout_mini_batch = config["training"]["rollout_mini_batch"]               
    method             = config["training"]["method"]                           
    clip_ratio_low     = config["training"]["clip_ratio_low"]                    
    clip_ratio_high    = config["training"]["clip_ratio_high"]

    use_entropy        = config["training"]["use_entropy"]  
    selected_percent   = config["training"]["selected_percent"]  
    use_TES_method     = config["training"]["use_TES_method"] 
    use_GMTS_method    = config["training"]["use_GMTS_method"] 

    read_model_path     = config["training"]["read_model_path"]                 
    doing_inverse       = config["training"]["doing_inverse"]
    kl_beta             = config["training"]["kl_beta"] 

    save_logp_gradient_path  = config["training"]["save_logp_gradient_path"]
    save_true_gradient_path  = config["training"]["save_true_gradient_path"]
    save_entrpy_path         = config["training"]["save_entrpy_path"]
    save_mask_path           = config["training"]["save_mask_path"]
    save_adv_pth             = config["training"]["save_adv_pth"]
    save_prefix_pth          = config["training"]["save_prefix_pth"]
    save_ids_path            = config["training"]["save_ids_path"]
    use_filter               = config["training"]["use_filter"]
    
    # data-setting:
    data_name          = config["data"]["data_name"]     
    old_model_device   = torch.device(config["model"]["old_model_device"])      
    
    pretrained_model_path = Path(config["model"]["pretrained_model_path"] )      
    device = torch.device(config["model"]["device"]) 
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config["model"]["dtype"], torch.bfloat16)
    torch.set_default_device(device)
    torch.random.manual_seed(config["training"]["random_seed"])

    BATCH_SIZE = config["training"]["batch_size"]                                                     
    NUM_QUESTIONS_PER_BATCH = config["training"]["num_questions_per_batch"]                           
    NUM_ANSWERS_PER_QUESTION = BATCH_SIZE // NUM_QUESTIONS_PER_BATCH                                 
    
    current_time = datetime.now().strftime(r"%Y%m%d-%H%M%S")
    tb_writer = SummaryWriter(log_dir=f"{config['training']['log_dir']}/{current_time}")
    tokenizer = Tokenizer(str(pretrained_model_path / "tokenizer.json"))
    
    train_dataset = mathTasksDataset(
        train_data_path=config["data"]["training_path"],
        test_data_path=config["data"]["testing_path"],
        tokenizer=tokenizer,
        split="train",
        data_name = data_name
    )

    generator = torch.Generator(device=device)

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn= mathTasksDataset.collate_fn,
        generator=generator,
        batch_size=NUM_QUESTIONS_PER_BATCH,
    )

    if(read_model_path==""):
        model = Transformer.from_pretrained(pretrained_model_path, device=device).train() # load model
    
    else: 
        model = Transformer.from_pretrained(pretrained_model_path, device=device).train() 
        state_dict = torch.load(read_model_path, map_location=device)                     
        model.load_state_dict(state_dict, strict=True) 
        start_steps = config["training"]["start_steps"] 
        del state_dict 
        torch.cuda.empty_cache()
    
    if(method=='grpo'):
        ref_model_device = torch.device(config["model"]["ref_model_device"]) 
        ref_model        = Transformer.from_pretrained(pretrained_model_path, device=ref_model_device)
    
    elif(method=='dapo'):
        ref_model_device = None
        ref_model  = None


    optimizer = MemoryEfficientAdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        betas=config["training"]["betas"],
        enabled=config["training"]["memory_efficient_adamw"],
    )

    start_time = time.time()
    ckpt_dir = Path(config["training"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    max_test_acc_pass1 = 0

    if(method=='grpo'):
        print("You are using the grpo method, no need to filter the q-a in the data.")
    
    if(method=='dapo'):
        print("You are using the dapo method, need to filter the q-a in the data.")

    for step, batch in enumerate(train_dataloader, start=1):
        episodes          = {}
        prefix_all        = batch.prefix
        prefix_tokens_all = batch.prefix_tokens
        prefix_token_ids  = batch.prefix_token_ids
        questions         = batch.questions
        answers           = batch.answers
        
        if(read_model_path!=""):
            if(step<start_steps):
                print("escape step" + str(step))
                continue
        
        for q in range(0, len(questions), rollout_mini_batch): 
            w = min(q + rollout_mini_batch, len(questions))
            batch_now = batch
            batch_now.prefix = prefix_all[q:w]
            batch_now.prefix_tokens = prefix_tokens_all[q:w]
            batch_now.prefix_token_ids = prefix_token_ids[q:w]
            batch_now.questions = questions[q:w]
            batch_now.answers = answers[q:w]

            # generating now.
            episodes_this_batch = rollout(model=model,
                                          tokenizer=tokenizer,
                                          batch=batch_now,
                                          max_gen_len=config["training"]["max_gen_len"],
                                          num_answer_per_question=NUM_ANSWERS_PER_QUESTION,
                                          reward_function=math_reward_function,
                                          device=device,
                                          dtype=dtype,
                                          data_name=data_name) 
            
            for item in episodes_this_batch:
                if(item.prefix not in episodes):
                    episodes[item.prefix]=[item]
                else:
                    episodes[item.prefix].append(item)
            
            if ((use_filter==True) and (method == 'dapo')):
                to_del = []
                for k, group in list(episodes.items()):  
                    if not group:  
                        continue
                    rewards = [item.reward for item in group]

                    uniq = set(rewards)
                    if uniq == {1} or uniq == {0}:
                        to_del.append(k)

                for k in to_del:
                    episodes.pop(k, None)

        results = update_policy(model=model,
                                optimizer=optimizer,
                                episodes=episodes,
                                mini_batch_size =  config["training"]["mini_batch_size"],
                                micro_batch_size = config["training"]["micro_batch_size"],
                                pad_token_id=tokenizer.pad_token_id,
                                max_grad_norm=config["training"]["max_grad_norm"],
                                device=device,
                                dtype=dtype,
                                method= method,
                                clip_ratio_low= clip_ratio_low,
                                clip_ratio_high= clip_ratio_high,
                                use_entropy= use_entropy,
                                use_TES_method= use_TES_method,
                                use_GMTS_method= use_GMTS_method,
                                doing_inverse       = doing_inverse,
                                ref_model           = ref_model,
                                ref_model_device    = ref_model_device,
                                old_model_device    = old_model_device,
                                kl_beta             = kl_beta,
                                selected_percent    =selected_percent)
        
        
        torch.cuda.synchronize()
        end_time = time.time()
        duration = end_time - start_time
        start_time = end_time

        if step % config["training"]["gradient_interval"] == 0:
            gradient(model=model,
                     episodes=episodes,
                     mini_batch_size =  config["training"]["mini_batch_size"],
                     micro_batch_size = config["training"]["micro_batch_size"],
                     pad_token_id=tokenizer.pad_token_id,
                     device=device,
                     dtype=dtype,
                     method                   = method,
                     clip_ratio_low           = clip_ratio_low,
                     clip_ratio_high          = clip_ratio_high,
                     save_entrpy_path         = save_entrpy_path + str(step) +"_entropy.pt",
                     save_logp_gradient_path  = save_logp_gradient_path + str(step) +"_logp_gradient.pt",
                     save_true_gradient_path  = save_true_gradient_path + str(step) +"_true_gradient.pt",
                     save_mask_path           = save_mask_path + str(step) +"_mask.pt",
                     save_adv_pth             = save_adv_pth + str(step) +"_adv.pt",
                     save_prefix_pth          = save_prefix_pth + str(step) +"_prefix.npy",
                     save_ids_path            = save_ids_path + str(step) +"_ids.pt",
                     ref_model                = ref_model,
                     ref_model_device         = ref_model_device,
                     old_model_device         = old_model_device,
                     kl_beta                  = kl_beta
            )

        # compute and log important metrics
        loss_list = results[0]
        grad_norm_list = results[1]
        entropy_list = results[2]

        reward = []
        formatted_reward = []
        answer_reward = []
        response_length = []
        
        for keys in episodes.keys():
            reward_keys           = [episode.reward for episode in episodes[keys]]
            formatted_reward_keys = [episode.reward_info["format_reward"] for episode in episodes[keys]]
            answer_reward_keys    = [episode.reward_info["answer_reward"] for episode in episodes[keys]]
            response_length_keys  = [len(episode.generated_token_ids) for episode in episodes[keys]]
            reward.extend(reward_keys)
            formatted_reward.extend(formatted_reward_keys)
            answer_reward.extend(answer_reward_keys)
            response_length.extend(response_length_keys)
        

        mean_reward           = np.mean(reward)
        std_reward            = np.std(reward)
        success_rate          = np.mean(answer_reward)
        format_reward         = np.mean(formatted_reward)
        mean_response_len     = np.mean(response_length)
        
        lr = optimizer.param_groups[0]["lr"]

        step_grad_norm       = np.mean(grad_norm_list)
        step_loss            = np.mean(loss_list)
        step_entropy         = np.mean(entropy_list)
        
        print(
            f"\rStep {step}, mean_reward: {mean_reward:.3f}, "
            f"train success_rate: {success_rate:.3f}, "
            f"grad_norm: {step_grad_norm:.3f}, duration: {duration:.3f}, "
            f"mean_response_len: {mean_response_len:.3f}, "
            f"entropy: {step_entropy:.3f}"
        )

        if step % config["training"]["eval_interval"] == 0:
            eval_success_rate_list, eval_average_token_length_list = evaluate(model, tokenizer, device, dtype, config)
            print(f"\rEval MATH-pass@1-success rate: {eval_success_rate_list[0]:.3f}" + " " * 100)
            tb_writer.add_scalar("MATH_success_rate/eval", eval_success_rate_list[0], 0)
            tb_writer.add_scalar("MATH_average_token_length/eval", eval_average_token_length_list[0], 0)
            

            if(max_test_acc_pass1 <= eval_success_rate_list[0]):
                max_test_acc_pass1 = eval_success_rate_list[0]
                output_file = ckpt_dir / f"ckpt_max_pass1.pt"
                torch.save(model.state_dict(), output_file)
                print(f"Saved checkpoint to {output_file}")
        
        # save checkpoint
        if step % config["training"]["ckpt_save_interval"] == 0:
            output_file = ckpt_dir / f"ckpt_{step:06d}.pt" 
            torch.save(model.state_dict(), output_file)
            print(f"Saved checkpoint to {output_file}")
        
        tb_writer.add_scalar("step_avg_loss", step_loss, step)
        tb_writer.add_scalar("step_mean_reward", mean_reward, step)
        tb_writer.add_scalar("step_std_reward", std_reward, step)
        tb_writer.add_scalar("step_success_rate/train", success_rate, step)
        tb_writer.add_scalar("step_format_reward", format_reward, step)
        tb_writer.add_scalar("step_avg_grad_norm", step_grad_norm, step)
        tb_writer.add_scalar("step_duration", duration, step)
        tb_writer.add_scalar("step_learning_rate", lr, step)
        tb_writer.add_scalar("step_mean_response_len", mean_response_len, step)
        tb_writer.add_scalar("step_avg_entropy", step_entropy, step)
        
        for i, episode in enumerate(episodes.values()):
            # TensorBoard treats text as markdown.
            for item in  episode:
                text = html.escape(item.text)
                tb_writer.add_text(f"text_{i}", f"<pre>{text}</pre>", step)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    main(args.config)