# main-method.
import dataclasses
import gc
import math
import numpy as np
import torch
import copy
from collections import defaultdict
from typing import Callable, List
from grpo_tools.data_types import Episode, MATH_MiniBatch
from grpo_tools.qwen2_model import Transformer
from grpo_tools.tokenizer import Tokenizer

# function1: approximate KL divergence used by GRPO, DAPO no need.
def approx_kl_divergence(log_probs, 
                         log_probs_ref, 
                         action_mask, 
                         batch_episodes,
                         group_number=16):
    
    """
    Monte-Carlo approximation of the KL divergence using the k3 estimator.
    Reference: http://joschu.net/blog/kl-approx.html

    Intended use:
      - Compute per-sample (or per-token) KL terms between a current policy and a
        reference/behavior policy in GRPO-style training.
      - Returns a vector `kl` aligned with `batch_episodes`.

    Expected shapes / types:
      - log_probs:     1D or token-level tensor aligned with batch entries (float).
      - log_probs_ref: same shape as `log_probs`.
      - action_mask:   same shape as `log_probs` (or broadcastable), with 1.0 for valid tokens and 0.0 for padding.
      - batch_episodes: iterable; len(batch_episodes) == len(log_probs) after any token aggregation you do.
        Each element must expose `.generated_token_ids` so we can read the token length |o_i| for GRPO object.

    """
    log_ratio = log_probs_ref.float() - log_probs.float()
    
    if action_mask is not None:
        log_ratio = log_ratio * action_mask
    
    kl = log_ratio.exp() - log_ratio - 1                      # estimate KL.
    
    assert len(kl)==len(batch_episodes)    

    batch_token_length = [len(item.generated_token_ids) for item in batch_episodes] # get |oi| for each answer.
        
    for i in range(len(kl)):
        kl[i] = kl[i]/(group_number*batch_token_length[i]) #  KL/(|oi|*G) in grpo object.

    return kl


# function2: GMTS-method.
def GMTS(tensor_list, 
         group_labels, 
         adv_list,
         frac,
         method='dapo',
         entropy_all_theta_prob = [],
         entropy_all_ref_prob = [],
         entropy_all_old_prob = [],
         clip_ratio_low = 0.2,
         clip_ratio_high = 0.28,
         beta = 0.001,
         group_number = 16,
         doing_inverse = False):
    
    """
    - tensor_list:            the entropy list for the pure answer part without padding and question.
    - group_labels:           the question list as the label for each answer.
    - adv_list:               the advantage for each Q-A-pair.
    - frac:                   selected ratio.
    - method:                 dapo or grpo.
    - entropy_all_theta_prob: \pi_theta list for the pure answer part without padding and question.
    - entropy_all_ref_prob:   \pi_ref   list for the pure answer part without padding and question.
    - entropy_all_old_prob:   \pi_old   list for the pure answer part without padding and question
    - beta                    kl-coef 
    - doing_inverse:          inverse-experiemt.
    """
    
    out = [t.clone() for t in tensor_list]                # clone the output.

    idx_by_group = defaultdict(list)
    
    for idx, g in enumerate(group_labels):
        idx_by_group[g].append(idx)                       # seperate into different groups, (idx_by_group[Q1] = [2,3,4,5] means 2345 from the same group) 

    for g, idxs in idx_by_group.items():                 
        entries = []
        
        assert len(idxs) == group_number                

        for tid in idxs:
            adv  = adv_list[tid]                          # get advantage.
            flat = tensor_list[tid].view(-1)              # get entropy.

            if(method=='grpo'):
                
                if(len(flat)!=0): 
                    kl_weighted = ((entropy_all_ref_prob[tid]-entropy_all_theta_prob[tid]).exp()*beta - beta)    # beta* \pi(ref)/\pi(theta) -beta
                    
                    kl_weighted = kl_weighted/(len(flat)*group_number)                                           #  KL/(|oi|*G)

                    theta_old_ratio = (entropy_all_theta_prob[tid] - entropy_all_old_prob[tid]).exp()            # pi_theta / pi_old  
                    
                    indecated_ratio = torch.full_like(theta_old_ratio, 1.0)                                      # I_trust
                    
                    mask1 = (theta_old_ratio > 1 + clip_ratio_high) & (adv > 0)
                    
                    indecated_ratio[mask1] = 0

                    mask2 = (theta_old_ratio < 1 - clip_ratio_low)  & (adv < 0)
                    
                    indecated_ratio[mask2] = 0 

                    ratio_summary = theta_old_ratio * indecated_ratio * adv                                      # ((pi_theta / pi_old) * I * A)/(|oi|*G)                                        

                    adv_grpo = ratio_summary + kl_weighted                                                  

                    # adv_grpo =  ( (pi_theta / pi_old) * I_trust * A  + beta * \pi(ref)/\pi(theta) - beta ) /  (|oi|*G)      

            elif(method=='dapo'):

                if(len(flat)!=0): 
                    
                    theta_old_ratio = (entropy_all_theta_prob[tid] - entropy_all_old_prob[tid]).exp()            # pi_theta / pi_old  
                    
                    indecated_ratio = torch.full_like(theta_old_ratio, 1.0)                                      # I_trust
                    
                    mask1 = (theta_old_ratio > 1 + clip_ratio_high) & (adv > 0)
                    
                    indecated_ratio[mask1] = 0

                    mask2 = (theta_old_ratio < 1 - clip_ratio_low)  & (adv < 0)
                    
                    indecated_ratio[mask2] = 0 

                    ratio_summary = theta_old_ratio * indecated_ratio * adv                                      # (pi_theta / pi_old) *I *A /(\sum_|oi|)  
                                      
                    adv_dapo = ratio_summary      

                    # adv_dapo =  ( (pi_theta / pi_old) * I_trust * A ) /  (\sum_|oi|)                                                         
            
            else:
                print("Your method is wrong, it can only be grpo or dapo.")
                assert False
            
            for eid, v in enumerate(flat):
                
                if(method=='dapo'):
                    entries.append((v.item() * abs(adv_dapo[eid]), tid, eid))      
                
                elif(method=='grpo'):
                    entries.append((v.item() * abs(adv_grpo[eid]), tid, eid))      
        
        
        entries.sort(key=lambda x: x[0])                                           # rank the score from low to high.
        
        k = max(1, math.ceil(len(entries) * frac))                                 # at least remove the min-one.
        
        if(doing_inverse==False):                                                  # doing the top selection.
            for _, tid, eid in entries[:k]:                                        
                out[tid].view(-1)[eid] = 0                                         # remove the bottom part, set their entropy as 0.
        
        elif(doing_inverse==True):                                                 # doing the bottom selection.
            for _, tid, eid in entries[k:]:  
                out[tid].view(-1)[eid] = 0                                         # remove the top part, set their entropy as 0.
    
    return out


# function3: rollout.
@torch.no_grad()
def rollout(
    model: Transformer, 
    batch,
    tokenizer: Tokenizer,
    max_gen_len: int,
    num_answer_per_question: int,
    reward_function: Callable,
    device: torch.device,
    dtype: torch.dtype,
    data_name : str,
    ) -> List[Episode]:

    end_token = tokenizer.eos_token                         # the end token
    end_token_id = tokenizer.eos_token_id                   # the end token id
    pad_token_id = tokenizer.pad_token_id                   # the pad token id
    prefix_token_ids = batch.prefix_token_ids               # question-prompt token id
    bsz = len(batch.prefix) * num_answer_per_question       # how many for total pair  (question number*group_size)
    min_prompt_len = min(len(t) for t in prefix_token_ids)  # the min-prompt-ids
    max_prompt_len = max(len(t) for t in prefix_token_ids)  # the max-promot-ids
    total_len = max_gen_len + max_prompt_len                # max-length-total

    model.init_kv_cache(max_batch_size=bsz,
                        max_seq_len=total_len,
                        device=device,
                        dtype=dtype)

    tokens = torch.full((bsz, total_len), pad_token_id, dtype=torch.long, device=device)

    for k, t in enumerate(prefix_token_ids):
        offset = k * num_answer_per_question
        for i in range(num_answer_per_question):
            tokens[offset + i, : len(t)] = torch.tensor(
                t, dtype=torch.long, device=device
            )

    prev_pos = 0
    input_text_mask = tokens != pad_token_id
    assert min_prompt_len < total_len
    is_finished = torch.zeros((bsz,), dtype=torch.bool, device=device)

    for cur_pos in range(min_prompt_len, total_len):
        print(
            f"\r* Generating trajectories: {cur_pos-min_prompt_len:>4d}/{total_len-min_prompt_len:>4d}",
            flush=True,
            end="",
        )
        with torch.autocast(device_type=device.type, dtype=dtype):
            logits = model.inference(tokens[:, prev_pos:cur_pos], prev_pos)
        probs = torch.softmax(logits[:, -1], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)                           #choose one from the multi-distribution.
        next_token = next_token.reshape(-1)
        next_token = torch.where(
            input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
        )

        # if an rollout is finished, we fill the rest of the tokens with pad_token_id
        next_token = torch.where(is_finished, pad_token_id, next_token)
        tokens[:, cur_pos] = next_token
        if end_token_id is not None:
            is_end_token = next_token == end_token_id
            is_generated_token = ~input_text_mask[:, cur_pos]
            is_finished = is_finished | (is_end_token & is_generated_token)
        prev_pos = cur_pos
        if is_finished.all():
            break
    
    model.del_kv_cache()
    gc.collect()
    torch.cuda.empty_cache()
    is_finished_list = is_finished.tolist()
    tokens_list = tokens.tolist()

    # prepare the output episodes
    episodes = []
    for i in range(bsz // num_answer_per_question):
        for j in range(num_answer_per_question):
            idx = i * num_answer_per_question + j
            generated_token_ids = tokens_list[idx][len(batch.prefix_token_ids[i]) :]
            # remove padding tokens
            if pad_token_id in generated_token_ids:
                generated_token_ids = generated_token_ids[: generated_token_ids.index(pad_token_id)]
            
            generated_text = tokenizer.detokenize(generated_token_ids)
            
            rewards = reward_function(
                response=generated_text,
                answers=batch.answers[i],
                end_token=end_token,
                data_name=data_name
            )
            
            # storage the dataset.
            episode = Episode(
                prefix = batch.prefix[i],                         # input
                text = batch.prefix[i] + generated_text,          # input+output
                prefix_token_ids = batch.prefix_token_ids[i],     # input_token_ids
                prefix_tokens = batch.prefix_tokens[i],           # input_token
                generated_token_ids = generated_token_ids,        # generate_token_ids
                is_finished = is_finished_list[idx],             
                reward = rewards["reward"],                       # reward
                reward_info = rewards["reward_info"],             # reward-details
            )
            episodes.append(episode)
    
    # clear the output line
    print("\r", end=" " * 100, flush=True)
    return episodes


# function4: normalized reward.
def normalize_rewards_per_group(episodes: List[Episode], method = 'grpo', group_number = 16) -> List[Episode]:

    """Normalize rewards per group. A group is defined by the prefix."""
    groups = defaultdict(list)
    for episode in episodes:
        groups[tuple(episode.prefix)].append(episode)                                           # seperate the different groups
    

    output = []
    for group in groups.values():
        group_rewards = [item.reward for item in group]                                         # ri in this group
        group_token_length = [len(item.generated_token_ids) for item in group]                  # |oi| in this group
        assert group_number == len(group_rewards)                              
        
        group_summary_token_number = sum(group_token_length)                                    # \sum_(|oi|)

        mean_reward = np.mean(group_rewards)      
        std_reward = np.std(group_rewards)

        for j in range(len(group)):
            episode = group[j]
            normalized_reward = (episode.reward - mean_reward) / (std_reward + 1e-6)            # get-adv
            
            if(method=='grpo'):
                if(group_token_length[j]!=0):
                    normalized_reward = (normalized_reward)/(group_token_length[j]*group_number) # A/(G*|oi|)
                else:
                    normalized_reward = 0
            
            elif(method=='dapo'):
                normalized_reward     = (normalized_reward)/group_summary_token_number            # A/\(sum_|oi|)
            
            episode = dataclasses.replace(episode, reward=normalized_reward)                      # replace the reward to the normalized_reward
            output.append(episode)

    return output


# function5: calculate the entropy.
def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.nn.functional.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probs * logits, dim=-1)
    return entropy


# function6: Updating the policy.
def update_policy(
    model,                      
    optimizer,
    episodes: dict,                   # ALL-dataset-size: (Batch-question) * group-number dict.
    mini_batch_size : int,            # updating question number per gradient steps.
    micro_batch_size: int,            
    pad_token_id: int,         
    max_grad_norm: float,
    device: torch.device,
    dtype: torch.dtype,
    method              = 'grpo',
    clip_ratio_low      = 0.2,
    clip_ratio_high     = 0.28,
    use_entropy         = False,      # whether use entropy, use will be GMTS/ETS 
    use_TES_method      = False,     
    use_GMTS_method     = False,  
    doing_inverse       = False,      # inverse selection?
    ref_model           = None,       
    ref_model_device    = None,       
    old_model_device    = None,       
    kl_beta             = 0.001,      # kl-weight
    selected_percent    = 1.0,        # selected_percent
):
    
    loss_summary = []     
    
    grad_norm_summary = []
    
    entropy_summary = []

    
    # step1: preparing the \pi_old and \pi_ref.
    #################################################################################################################
    assert old_model_device!=None                                # must has device for \pi_old-model
    
    device_old = torch.device(old_model_device)                  # get old_device.

    if(method=='grpo'):
        assert ref_model!=None
        assert ref_model_device!=None
        device_ref = torch.device(ref_model_device)              # get device.
    
    
    old_model = copy.deepcopy(model).to(device_old)              # old model
    #################################################################################################################
    
    all_keys = list(episodes.keys())  # question's number

    for updating_epoch in range(0, len(episodes.keys()), mini_batch_size):

        updating_questions = all_keys[updating_epoch: updating_epoch+mini_batch_size] # get these questions.

        episodes_updating = []

        for question in updating_questions:
            episodes_updating.extend(episodes[question]) # get the updating data.
        
        
        # step2: calculate the advantage and norm it.
        #################################################################################################################
        episodes_updating = normalize_rewards_per_group(episodes_updating, method)
        #################################################################################################################

        num_target_tokens = sum(len(episode.generated_token_ids) for episode in episodes_updating)   # sum(token)
        
        entropy_all = 0.0                                                 

        # step3: sampling and collecting
        #################################################################################################################
        batch_target_masks_all = []                                     
        batch_target_entropy_all = []                                 
        batch_target_advantages_all = []                                  
        batch_target_prefix = []                 
        batch_target_ref_prob = []                                 
        batch_target_theta_prob = []         
        batch_target_old_prob = []           


        for i in range(0, len(episodes_updating), micro_batch_size):
            print(f"\r* Computing the all entropy here: {i:>2d}/{len(episodes_updating):>2d}",flush=True, end="")

            j = min(i + micro_batch_size, len(episodes_updating))
            
            batch_episodes  = episodes_updating[i:j]           # get the calculating data.

            batch_length    = [len(episode.prefix_token_ids) + len(episode.generated_token_ids) for episode in batch_episodes] # get each q-a-length.
            
            batch_lenth_max = max(batch_length)                 # get the max of it
        
            batch_token_ids = [episode.prefix_token_ids + episode.generated_token_ids + [pad_token_id] * (batch_lenth_max - batch_length[i]) for i, episode in enumerate(batch_episodes)]                # right padding.
            
            batch_masks =     [[0] * len(episode.prefix_token_ids) + [1] * len(episode.generated_token_ids) + [0] * (batch_lenth_max - batch_length[i]) for i, episode in enumerate(batch_episodes)]     # right padding.                 
            
            batch_advantages = [episode.reward for episode in batch_episodes]                   
            
            batch_prefix     = [episode.prefix for episode in batch_episodes]                   
            
            # turn into torch type.
            batch_token_ids  = torch.tensor(batch_token_ids, device=device, dtype=torch.long)
            batch_masks      = torch.tensor(batch_masks, device=device, dtype=torch.bool)
            batch_advantages = torch.tensor(batch_advantages, device=device, dtype=torch.float32)
            
            with torch.autocast(device_type=device.type, dtype=dtype):
                input_token_ids =  batch_token_ids[:, :-1] 
                target_token_ids = batch_token_ids[:, 1:]   
                target_masks = batch_masks[:, 1:]           
                target_adv = batch_advantages
            

            with torch.no_grad():

                logits = model.forward(input_token_ids).float()                                   # get the logits.
                
                log_probs = -torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    target_token_ids.reshape(-1),
                    ignore_index=pad_token_id,
                    reduction="none").reshape(input_token_ids.shape[0], -1)                       # get the logps
                
                old_input_token_ids = input_token_ids.to(device_old)
                
                old_logits = old_model.forward(old_input_token_ids).float().to(device)            

                old_log_probs = -torch.nn.functional.cross_entropy(
                    old_logits.reshape(-1, old_logits.size(-1)),
                    target_token_ids.reshape(-1),
                    ignore_index=pad_token_id,
                    reduction="none").reshape(input_token_ids.shape[0], -1)                       
                

                if(method=='grpo'):
                    ref_input_token_ids = input_token_ids.to(device_ref)
                    
                    ref_logits = ref_model.forward(ref_input_token_ids).float().to(device)
                    
                    ref_log_probs = -torch.nn.functional.cross_entropy(
                        ref_logits.reshape(-1, ref_logits.size(-1)),
                        target_token_ids.reshape(-1),
                        ignore_index=pad_token_id,
                        reduction="none").reshape(input_token_ids.shape[0], -1)                      # get ref-logp
                    
                    # calculating finished, storage.
                    batch_target_ref_prob.append(ref_log_probs)
                
                token_entropy = compute_entropy(logits)                                              # get entropy

                entropy_all = entropy_all + (token_entropy * target_masks).sum() / num_target_tokens 
            
            batch_target_theta_prob.append(log_probs)
            batch_target_old_prob.append(old_log_probs)      
            batch_target_masks_all.append(target_masks)    
            batch_target_advantages_all.append(target_adv) 
            batch_target_entropy_all.append(token_entropy) 
            batch_target_prefix.append(batch_prefix)

            del batch_episodes, batch_length, batch_lenth_max, batch_token_ids, batch_masks, batch_advantages, batch_prefix, 
            del input_token_ids, target_token_ids, target_masks, target_adv, 
            del logits, token_entropy, log_probs, old_logits, old_log_probs
            
            if(method=='grpo'):
                del ref_log_probs, ref_logits
            
            torch.cuda.empty_cache()
        #################################################################################################################

        if(use_entropy==True):      
            
            entropy_number              = []      # entropy's number
            entropy_all_index           = []      # entropy's start id and end id     
            entropy_all_group_prefix    = []      # prefix for each answer
            entropy_all_adv             = []      # advantage
            entropy_all_theta_prob      = []      
            entropy_all_ref_prob        = []      
            entropy_all_old_prob        = []      

            for k in range(len(batch_target_entropy_all)):

                entorpy_batch       = batch_target_entropy_all[k]     
                mask_batch          = batch_target_masks_all[k]       
                adv_batch           = batch_target_advantages_all[k]  
                prefix_batch        = batch_target_prefix[k]          
                theta_prob_batch    = batch_target_theta_prob[k]      
                old_prob_batch      = batch_target_old_prob[k]        
                
                if(method=='grpo'):
                    ref_prob_batch  = batch_target_ref_prob[k]        
                
                for l in range(len(mask_batch)):
                    entropy = entorpy_batch[l]                        
                    mask = mask_batch[l]                              
                    prefix_now = prefix_batch[l]                      
                    adv_now = adv_batch[l]
                    theta_prob_now = theta_prob_batch[l]
                    old_prob_now = old_prob_batch[l]
                    
                    if(method=='grpo'):
                        ref_prob_now = ref_prob_batch[l]


                    padded = torch.cat([torch.tensor([False]), mask, torch.tensor([False])])
                    changes = padded[1:] != padded[:-1]
                    change_idx = torch.nonzero(changes).flatten()
                    
                    if(len(change_idx)<2):                               
                        entropy_number.append(torch.tensor([]))         
                        entropy_all_index.append(torch.tensor([-1, -1])) 
                        entropy_all_theta_prob.append(torch.tensor([]))
                        entropy_all_old_prob.append(torch.tensor([]))
                        if(method=='grpo'):
                            entropy_all_ref_prob.append(torch.tensor([])) # ref-logp
                    
                    else:
                        True_start_id =  change_idx[0::2].item()                      # get the start ids
                        True_end_id   = (change_idx[1::2] - 1).item()                 # get the end ids
                        entropy_number.append(entropy[True_start_id:True_end_id+1])   # entropy for calculating
                        entropy_all_index.append([True_start_id,True_end_id])         # start and end
                        entropy_all_theta_prob.append(theta_prob_now[True_start_id:True_end_id+1]) # theta-logp
                        entropy_all_old_prob.append(old_prob_now[True_start_id:True_end_id+1])     # old-logp
                        
                        if(method=='grpo'):
                            entropy_all_ref_prob.append(ref_prob_now[True_start_id:True_end_id+1]) # ref-logp
                    
                    entropy_all_group_prefix.append(prefix_now)                                    # label-index
                    entropy_all_adv.append(adv_now)                                                # adv
            


            if((use_TES_method==True) and (use_GMTS_method==False)):        # clip entropy overall.
                
                all_elements = torch.cat(entropy_number)                    # cat all entropy
                num_to_remove = int(len(all_elements) * selected_percent)   # remove number_size.

                if(selected_percent!=0):
                    sorted_elements, sorted_indices = torch.sort(all_elements)   # sorted from low to high
                    threshold_value = sorted_elements[num_to_remove - 1]         # threshold_value
                else:
                    threshold_value = 0.0
                

                for g in range(len(entropy_number)): 
                    
                    if(doing_inverse==False):
                        entropy_number[g] = torch.where(entropy_number[g] < threshold_value, torch.tensor(0), entropy_number[g]) 
                    
                    elif(doing_inverse==True):
                        entropy_number[g] = torch.where(entropy_number[g] > threshold_value, torch.tensor(0), entropy_number[g]) 
            

            
    
            elif ((use_GMTS_method==True) and (use_TES_method==False)):
                entropy_number = GMTS(entropy_number, 
                                      entropy_all_group_prefix, 
                                      entropy_all_adv,
                                      selected_percent,
                                      method,
                                      entropy_all_theta_prob,
                                      entropy_all_ref_prob,
                                      entropy_all_old_prob,
                                      clip_ratio_low,
                                      clip_ratio_high,
                                      kl_beta ,
                                      group_number=16,
                                      doing_inverse=doing_inverse)


        del batch_target_masks_all, batch_target_entropy_all
        del batch_target_prefix, batch_target_theta_prob

        if(use_entropy==True):
            del entropy_all_group_prefix, entropy_all_adv, entropy_all_theta_prob, entropy_all_ref_prob, entropy_all_old_prob
        
        torch.cuda.empty_cache()
        
        number_now = -1

        for i in range(0, len(episodes_updating), micro_batch_size): 
            print(f"\r* Updating the policy now: {i:>2d}/{len(episodes_updating):>2d}",flush=True,end="")
            
            number_now +=1 

            j = min(i + micro_batch_size, len(episodes_updating))
            
            batch_episodes = episodes_updating[i:j] 
            
            batch_length = [len(episode.prefix_token_ids) + len(episode.generated_token_ids) for episode in batch_episodes] 

            batch_lenth_max = max(batch_length)
        
            batch_token_ids = [episode.prefix_token_ids + episode.generated_token_ids + [pad_token_id] * (batch_lenth_max - batch_length[i]) for i, episode in enumerate(batch_episodes)] 
            
            batch_masks     = [[0] * len(episode.prefix_token_ids) + [1] * len(episode.generated_token_ids) + [0] * (batch_lenth_max - batch_length[i]) for i, episode in enumerate(batch_episodes)]
            
            batch_advantages = [episode.reward for episode in batch_episodes] 
            
            batch_token_ids = torch.tensor(batch_token_ids, device=device, dtype=torch.long)
            batch_masks = torch.tensor(batch_masks, device=device, dtype=torch.bool)
            batch_advantages = torch.tensor(batch_advantages, device=device, dtype=torch.float32)
            
            with torch.autocast(device_type=device.type, dtype=dtype):
                input_token_ids =  batch_token_ids[:, :-1]
                target_token_ids = batch_token_ids[:, 1:] 
                target_masks = batch_masks[:, 1:]         
                target_adv = batch_advantages
            
            # cal logp with gradient.
            logits = model.forward(input_token_ids).float()
                
            log_probs = -torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_token_ids.reshape(-1),
                ignore_index=pad_token_id,
                reduction="none",
            ).reshape(input_token_ids.shape[0], -1) 

            old_log_probs     = batch_target_old_prob[number_now].detach() # get the old-logps

            if(method=='grpo'):
                ref_log_probs = batch_target_ref_prob[number_now].detach() 

            ratio = (log_probs - old_log_probs.detach()).exp()            # ratio

            # dapo/grpo
            if(use_entropy==False):
                surr1 = ratio * batch_target_advantages_all[number_now][:, None]
                surr2 = ratio.clamp(1 - clip_ratio_low,  1 + clip_ratio_high) * batch_target_advantages_all[number_now][:, None]

                if(method=='dapo'): 
                    obj = torch.min(surr1, surr2)
                    obj = (obj * target_masks).sum()
                    loss = -obj
                    loss.backward()
                
                elif(method=='grpo'): 
                    kl = approx_kl_divergence(
                        log_probs=log_probs,
                        log_probs_ref=ref_log_probs,
                        action_mask=target_masks,
                        batch_episodes=batch_episodes,
                    )

                    obj = -torch.min(surr1, surr2) + kl_beta *kl
                    obj = (obj * target_masks).sum()
                    loss = obj
                    loss.backward()
                
                else:
                    assert False

                del batch_episodes, batch_length, batch_lenth_max, batch_token_ids, batch_masks, batch_advantages, 
                del input_token_ids, target_token_ids, target_masks, target_adv
                del logits, log_probs, ratio, obj, surr1, surr2, old_log_probs
                
                if(method=='grpo'):
                    del ref_log_probs, kl
                
                torch.cuda.empty_cache()

            elif(use_entropy==True):
                
                mask_batch    = target_masks                      
                adv_batch     = batch_target_advantages_all[number_now]                              
                target_adv    = torch.zeros_like(mask_batch, dtype=torch.float32)                    

                for l in range(len(mask_batch)):
                    [True_start_id,True_end_id] = entropy_all_index[number_now*len(mask_batch) + l]   
                    now_entropy                 = entropy_number[number_now*len(mask_batch) + l]      
                    
                    if((True_start_id==-1) and (True_end_id==-1)):
                        assert len(now_entropy)==0                                                    
                        continue

                    cal_fit  = [1 if item!=0 else 0  for item in now_entropy]                         

                    adv_list = [adv_batch[l]*item for item in cal_fit]                                

                    adv_list = torch.tensor(adv_list).to(device)
                    target_adv[l][True_start_id:True_end_id+1] = adv_list                             

                surr1 = ratio * target_adv
                surr2 = ratio.clamp(1 - clip_ratio_low, 1 + clip_ratio_high) * target_adv

                if(method=='dapo'):
                    obj = torch.min(surr1, surr2)
                    obj = (obj * mask_batch).sum()
                    loss = -obj
                    loss.backward()
                

                elif(method=='grpo'):
                    kl = approx_kl_divergence(
                        log_probs=log_probs,
                        log_probs_ref=ref_log_probs,
                        action_mask=mask_batch,
                        batch_episodes=batch_episodes,
                    )
                    obj = -torch.min(surr1, surr2) + kl_beta *kl
                    obj = (obj * mask_batch).sum()
                    loss = obj
                    loss.backward()
                
                else:
                    assert False

                del batch_episodes, batch_length, batch_lenth_max, batch_token_ids, batch_masks, batch_advantages
                del input_token_ids, target_token_ids, target_masks, target_adv
                del logits, log_probs, ratio, obj, surr1, surr2, old_log_probs
                
                if(method=='grpo'):
                    del ref_log_probs, kl
                
                torch.cuda.empty_cache()
        
        torch.cuda.empty_cache()
        
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        
        optimizer.step()
        
        optimizer.zero_grad(set_to_none=True)
        
        loss_summary.append(loss.item())
        
        grad_norm_summary.append(grad_norm.item())
        
        entropy_summary.append(entropy_all.item())

    return [loss_summary, grad_norm_summary, entropy_summary]


# function7: Testing the gradient, logp and true gradient.
def gradient(
    model,
    episodes: dict,
    mini_batch_size : int,        
    micro_batch_size: int,        
    pad_token_id: int,
    device: torch.device,
    dtype: torch.dtype,
    method         = 'grpo',
    clip_ratio_low  = 0.2,
    clip_ratio_high = 0.28,
    save_entrpy_path = '',           
    save_logp_gradient_path   = '',   
    save_true_gradient_path   = '',
    save_mask_path   = '',
    save_adv_pth     = '',
    save_prefix_pth  = '',
    save_ids_path    = '',
    ref_model        = None,   
    ref_model_device = None,   
    old_model_device = None,   
    kl_beta          = 0.001,  
):

    assert old_model_device!=None                                
    
    device_old = torch.device(old_model_device)                 

    if(method=='grpo'):
        assert ref_model!=None
        assert ref_model_device!=None
        device_ref = torch.device(ref_model_device)              
    
    old_model = copy.deepcopy(model).to(device_old)              

    all_keys = list(episodes.keys())  # question's number

    # only need to save the first mini-batch-size :

    updating_questions = all_keys[0: 0 + 10] # get these questions.

    episodes_updating = []

    for question in updating_questions:
        episodes_updating.extend(episodes[question]) # get the updating data.

    episodes_updating = normalize_rewards_per_group(episodes_updating, method)
        
    batch_target_masks_all = []         # save the mask
    batch_target_advantages_all = []    # save the adv.
    batch_target_prefix = []            # save the input_prefix
    batch_target_ids_all = []

    batch_target_entropy_all = []       # save the entropy
    batch_target_logp_gradient_all = [] # save the logp_gradient
    batch_target_true_gradient_all = [] # save the true_gradient

    for i in range(0, len(episodes_updating), micro_batch_size): 
        print(f"\r* Doing calcuting the gradient now: {i:>2d}/{len(episodes_updating):>2d}", flush=True, end="")
        j = min(i + micro_batch_size, len(episodes_updating))

        batch_episodes = episodes_updating[i:j]

        batch_length = [len(episode.prefix_token_ids) + len(episode.generated_token_ids)for episode in batch_episodes] 

        batch_lenth_max = max(batch_length)
    
        batch_token_ids = [episode.prefix_token_ids + episode.generated_token_ids + [pad_token_id] * (batch_lenth_max - batch_length[i]) for i, episode in enumerate(batch_episodes)] 
        
        batch_masks =    [[0] * len(episode.prefix_token_ids) + [1] * len(episode.generated_token_ids) + [0] * (batch_lenth_max - batch_length[i]) for i, episode in enumerate(batch_episodes)]
        
        batch_advantages = [episode.reward for episode in batch_episodes] 
        
        batch_prefix     = [episode.prefix for episode in batch_episodes] 

        batch_target_prefix.append(batch_prefix)                          

        batch_token_ids = torch.tensor(batch_token_ids, device=device, dtype=torch.long)
        batch_masks = torch.tensor(batch_masks, device=device, dtype=torch.bool)
        batch_advantages = torch.tensor(batch_advantages, device=device, dtype=torch.float32)
        
        with torch.autocast(device_type=device.type, dtype=dtype):
            input_token_ids =  batch_token_ids[:, :-1]
            target_token_ids = batch_token_ids[:, 1:] 
            target_masks = batch_masks[:, 1:]         
            target_adv = batch_advantages
            logits = model.forward(input_token_ids).float()
        
        log_probs = -torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            target_token_ids.reshape(-1),
            ignore_index=pad_token_id,
            reduction="none",
        ).reshape(input_token_ids.shape[0], -1) 


        with torch.no_grad():
        
            old_input_token_ids = input_token_ids.to(device_old)
            old_logits = old_model.forward(old_input_token_ids).float().to(device)            

            old_log_probs = -torch.nn.functional.cross_entropy(
                old_logits.reshape(-1, old_logits.size(-1)),
                target_token_ids.reshape(-1),
                ignore_index=pad_token_id,
                reduction="none").reshape(input_token_ids.shape[0], -1)                       
                

            if(method=='grpo'):
                ref_input_token_ids = input_token_ids.to(device_ref)
                ref_logits = ref_model.forward(ref_input_token_ids).float().to(device)
                    
                ref_log_probs = -torch.nn.functional.cross_entropy(
                    ref_logits.reshape(-1, ref_logits.size(-1)),
                    target_token_ids.reshape(-1),
                    ignore_index=pad_token_id,
                    reduction="none").reshape(input_token_ids.shape[0], -1)
            
            token_entropy = compute_entropy(logits)    


        ratio = (log_probs - old_log_probs.detach()).exp()           # ratio # p/p_old
        surr1 = ratio
        surr2 = ratio.clamp(1 - clip_ratio_low, 1 + clip_ratio_high)
        obj = torch.min(surr1, surr2)
        obj = (obj * target_masks) 
        batch_gradient = []         
        for k1 in range(len(obj)): 
            true_norm = []
            for k2 in range(len(obj[k1])):
                loss = -obj[k1][k2]
                grads = torch.autograd.grad(loss,model.parameters(), retain_graph=True)
                norm = torch.linalg.vector_norm(torch.cat([g.reshape(-1) for g in grads]))
                true_norm.append(norm)
            batch_gradient.append(true_norm)
        


        surr3 = ratio * target_adv[:, None]
        surr4 = ratio.clamp(1 - clip_ratio_low, 1 + clip_ratio_high) * target_adv[:, None]

        if(method=='dapo'): 
            obj_new = torch.min(surr3, surr4)
            obj_new = (obj_new * target_masks)

            batch_gradient_true = [] 

            for k1 in range(len(obj_new)): 
                true_norm = []
                for k2 in range(len(obj_new[k1])):
                    loss = -obj_new[k1][k2]
                    grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True)
                    norm = torch.linalg.vector_norm(torch.cat([g.reshape(-1) for g in grads]))
                    true_norm.append(norm)
                batch_gradient_true.append(true_norm)
        

        elif(method=='grpo'):
            kl = approx_kl_divergence(
                log_probs=log_probs,
                log_probs_ref=ref_log_probs,
                action_mask=target_masks,
                batch_episodes=batch_episodes,
            )
            obj_new = torch.min(surr3, surr4) - kl_beta *kl
            obj_new = (obj_new * target_masks)

            batch_gradient_true = [] 

            for k1 in range(len(obj_new)): 
                true_norm = []
                for k2 in range(len(obj_new[k1])):
                    loss = -obj_new[k1][k2]
                    grads = torch.autograd.grad(loss, model.parameters(), retain_graph=True)
                    norm = torch.linalg.vector_norm(torch.cat([g.reshape(-1) for g in grads]))
                    true_norm.append(norm)
                batch_gradient_true.append(true_norm)

        batch_target_logp_gradient_all.append(batch_gradient)        # logp gradient
        batch_target_true_gradient_all.append(batch_gradient_true)   # true gradient
        batch_target_masks_all.append(target_masks)                  # mask
        batch_target_advantages_all.append(target_adv)               # adv
        batch_target_entropy_all.append(token_entropy)               # entropy
        batch_target_ids_all.append(target_token_ids)

        del batch_gradient, batch_gradient_true, target_masks, target_adv, token_entropy, target_token_ids
        torch.cuda.empty_cache()

    torch.save(batch_target_masks_all,  save_mask_path)     
    torch.save(batch_target_entropy_all,  save_entrpy_path) 
    torch.save(batch_target_advantages_all, save_adv_pth)   
    torch.save(batch_target_logp_gradient_all, save_logp_gradient_path)   
    torch.save(batch_target_true_gradient_all,  save_true_gradient_path)
    torch.save(batch_target_ids_all,            save_ids_path)

    string_array = np.array(batch_target_prefix)
    np.save(save_prefix_pth, string_array)