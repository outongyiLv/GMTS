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





