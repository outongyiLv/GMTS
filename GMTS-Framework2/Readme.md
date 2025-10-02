# GMTS-Framework2

> The framework below is built on verl and reflects our modifications and the associated execution workflow.

## 🔄 Changing (1) Data reordering

We observed that, when verl enters parallel training, it does not strictly proceed group by group. Instead, it typically computes the advantage jointly and then mixes the data for gradient computation. As a first step, we implemented data reordering in verl/trainer/ppo/ray_trainer.py:

```python
    def reorganize_batch_by_uid(self, batch):
        import copy
        reorganized = copy.deepcopy(batch) # copy

        question_uids = batch.non_tensor_batch["uid"]
        assert len(set(question_uids))==self.config.data.train_batch_size 
        
        group_indices = defaultdict(list)
        for idx, uid in enumerate(question_uids):
            group_indices[uid].append(idx)
        
        assert all(len(indices) == self.config.actor_rollout_ref.rollout.n for indices in group_indices.values())

        batch_batch_keys = batch.batch.keys()
        batch_non_tensor_batch_keys = batch.non_tensor_batch.keys()
        batch_meta_info_keys = batch.meta_info.keys()

        for keys in batch_batch_keys:
            data = batch.batch[keys] 
            processed_data = []
            
            for group_keys in group_indices.keys():
                group_list = group_indices[group_keys] 
                group_data = data[group_list]
                processed_data.append(group_data)
            
            processed_data = torch.cat(processed_data, dim=0)
            
            reorganized.batch[keys] = processed_data 
        
        for keys in batch_non_tensor_batch_keys:
            
            data = batch.non_tensor_batch[keys]
            processed_data = []
            
            for group_keys in group_indices.keys():
                group_list = group_indices[group_keys] 
                group_data = data[group_list]
                processed_data.append(group_data)

            processed_data = np.concatenate(processed_data, axis=0)  
            reorganized.non_tensor_batch[keys] = processed_data 
        
        for keys in batch_meta_info_keys:
            
            if(keys!='temperature'):
                data = batch.meta_info[keys]
                processed_data = []
                
                for group_keys in group_indices.keys():
                    group_list = group_indices[group_keys] 
                    group_data = [data[item] for item in group_list]
                    processed_data.extend(group_data)
            
                reorganized.meta_info[keys] = processed_data 
            
            else:
                continue
        
        return reorganized
```
and doing this after calculating the advantages:
```python
batch = self.reorganize_batch_by_uid(batch) # ranking the batch
```

## 🔄 Changing (2) GMTS selection
We do the GTMS and ETS main part in verl/workers/actor/dp_actor.py, update_policy:

```python
                mini_batch = data 

                if ((doing_entropy_clipping_type=="ETS") or (doing_entropy_clipping_type=="GMTS")) :

                    # step1: separate into micro-batch to calculate entropy and logp
                    if has_multi_modal_inputs:
                        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                        num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                        micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                    elif self.config.use_dynamic_bsz:
                        max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                        micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len) # auto
                    else:
                        self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                        # split batch into micro_batches
                        micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                    # step2:calculating
                    all_entropy_now = []
                    logp = []
                    
                    for data in micro_batches:
                        if isinstance(data, DataProto):
                            data = {**data.batch.to(torch.cuda.current_device()), **data.non_tensor_batch}
                        else:
                            data = data.to(torch.cuda.current_device())  # actor device is cpu when using offload
                        
                        with torch.no_grad(): # no grad
                            micro_entropy, micro_logp = self._forward_micro_batch(micro_batch=data, temperature=temperature)

                        all_entropy_now.append(micro_entropy)
                        logp.append(micro_logp)
                    
                    all_entropy_now = torch.cat(all_entropy_now, dim=0)  
                    logp            = torch.cat(logp, dim=0)

                    if(doing_entropy_clipping_type=="ETS"):
                        all_entropy_now_flatten = all_entropy_now.flatten()  # flatten all entropy
                        non_zero_all_entropy_now_flatten = all_entropy_now_flatten[all_entropy_now_flatten!=0] # remove the entropy equal zero (padding) 
                        sorted_elements, sorted_indices = torch.sort(non_zero_all_entropy_now_flatten) # sorted
                    
                        if(doing_entropy_clipping_percent==0.0):
                            clipping_threshold_value = 0.0
                        
                        elif(doing_entropy_clipping_percent==1.0):
                            print("can not do this, because you clip all the data here.")
                            assert False
                        
                        else:
                            num_to_remove = int(len(non_zero_all_entropy_now_flatten) * doing_entropy_clipping_percent) 
                            clipping_threshold_value = sorted_elements[num_to_remove - 1]  

                        mini_batch['advantages'] = torch.where(all_entropy_now <= clipping_threshold_value, torch.tensor(0), mini_batch['advantages']) 

                    elif(doing_entropy_clipping_type=="GMTS"): using GMTS

                        clip_ratio      = self.config.clip_ratio
                        clip_ratio_low  = self.config.clip_ratio_low if self.config.clip_ratio_low is not None else clip_ratio
                        clip_ratio_high = self.config.clip_ratio_high if self.config.clip_ratio_high is not None else clip_ratio    

                        theta_old_ratio = (logp - mini_batch['old_log_probs']).exp()  # pi_theta / pi_old 

                        adv = mini_batch['advantages']                                # advantange
                        
                        indecated_ratio = torch.full_like(theta_old_ratio, 1.0)       # I_trust

                        mask1 = (theta_old_ratio > 1 + clip_ratio_high) & (adv > 0)
                        indecated_ratio[mask1] = 0

                        mask2 = (theta_old_ratio < 1 - clip_ratio_low) & (adv < 0)
                        indecated_ratio[mask2] = 0                                   

                        ratio_summary = theta_old_ratio * adv * indecated_ratio

                        # I_trust*pi_theta / pi_old * advantange 

                        if self.config.use_kl_loss: # using kl loss
                            ref_theta_ratio = (mini_batch['ref_log_prob'] - logp).exp() # pi_ref / pi_theta
                            kl_loss_coef = self.config.kl_loss_coef
                            ratio_summary = ratio_summary + kl_loss_coef * ref_theta_ratio - kl_loss_coef # add kl-div and do the abs here.
                        
                        # I_trust * (pi_theta / pi_old) * advantange + kl_loss_coef * (pi_ref / pi_theta) - kl_loss_coef
                        ratio_summary = abs(ratio_summary) 

                        assert all_entropy_now.size()[0]%16==0 # group equals to 16
                        number_question = all_entropy_now.size()[0]//16 
                        print(number_question)

                        for i in range(0, number_question, 16):
                            all_entropy_now_group = all_entropy_now[i:i+16]
                            ratio_summary_group = ratio_summary[i:i+16]
                            
                            all_entropy_now_adv_group = all_entropy_now_group * ratio_summary_group  
                            all_entropy_now_flatten_adv_group = all_entropy_now_adv_group.flatten() 

                            non_zero_all_entropy_now_flatten_adv_group = all_entropy_now_flatten_adv_group[all_entropy_now_flatten_adv_group!=0]

                            sorted_elements, sorted_indices = torch.sort(non_zero_all_entropy_now_flatten_adv_group) 
                            
                            if(doing_entropy_clipping_percent==0.0):
                                clipping_threshold_value_group = 0.0
                            
                            elif(doing_entropy_clipping_percent==1.0):
                                print("can not do this, because you clip all the data here.")
                                assert False
                            
                            else:
                                num_to_remove = int(len(non_zero_all_entropy_now_flatten_adv_group) * doing_entropy_clipping_percent) 
                                
                                if(num_to_remove==0):
                                    clipping_threshold_value_group = 0
                                else:
                                    clipping_threshold_value_group = sorted_elements[num_to_remove - 1]  

                            mini_batch['advantages'][i:i+16] = torch.where(all_entropy_now_adv_group <= clipping_threshold_value_group, torch.tensor(0), mini_batch['advantages'][i:i+16])

```

## 🚀 Start

### 1) Environment Setup

Please follow verl's environment setup in this [part](https://github.com/volcengine/verl/blob/main/scripts/install_vllm_sglang_mcore.sh)

### 2) Running GMTS on Qwen3-8B

```bash
bash examples/dapo_trainer/run_qwen3_8b_qwa.sh
```