# doing evalution 
import torch
import numpy as np
from grpo_tools.task_type.math_task_QW import mathTasksDataset, math_reward_function
from grpo_tools.grpo import rollout
from grpo_tools.qwen2_model import Transformer           
from grpo_tools.tokenizer import Tokenizer         
from torch.utils.data import DataLoader

def evaluate(model, 
             tokenizer, 
             device, 
             dtype, 
             train_data_path,
             test_data_path,
             data_name,
             batch_size,
             max_gen_len = 2048,
             ):

    test_dataset = mathTasksDataset(train_data_path=train_data_path,
                                    test_data_path=test_data_path,
                                    tokenizer=tokenizer,
                                    split="test",
                                    data_name = data_name)
    
    generator = torch.Generator(device="cpu")

    dataloader = DataLoader(
            test_dataset,
            shuffle=False,
            collate_fn=mathTasksDataset.collate_fn,
            generator=generator,
            batch_size = batch_size,
            drop_last=False,
        )
    
    success = []
    token_length = []
    
    for batch in dataloader:
        episodes = rollout(model=model,
                           tokenizer=tokenizer,
                           batch=batch,
                           max_gen_len=max_gen_len,
                           num_answer_per_question=1,
                           reward_function=math_reward_function,
                           device=device,
                           dtype=dtype,
                           data_name=data_name)
        
        success.extend([episode.reward_info["answer_reward"] for episode in episodes])
        token_length.extend([len(episode.generated_token_ids) for episode in episodes])
    
    return np.mean(success), np.mean(token_length)


if __name__ == '__main__':
    device_number = "cuda:0"
    
    average_number = 16

    device = torch.device(device_number) 

    model_struct = "./GMTS/model/Qwen2.5-Math-1.5B" # base model struct.

    saving_path = "./GMTS/GMTS-Framework1/result_evaluation/Qwen2.5-math-1.5b-grpo-ETS"

    saving_path_list = [saving_path + "/result_avg" + str(i+1) + ".npy" for i in range(average_number)]

    evaluation_batch = 128

    evalution_data_list = ["./GMTS/GMTS-Framework1/test_data/AIME_2024",
                           "./GMTS/GMTS-Framework1/test_data/amc23",
                           "./GMTS/GMTS-Framework1/test_data/math-500",
                           "./GMTS/GMTS-Framework1/test_data/minerva",
                           "./GMTS/GMTS-Framework1/test_data/OlympiadBench"]
    
    evalution_model_weight_list = ["/home/zyw/GMTS/GMTS-Framework1/result/ckpt_dir/Qwen2.5-math-1.5b-grpo-ETS/ckpt_max_pass1.pt"]
    
    evalution_data_name_list = ["aime",
                                "amc",
                                "math",
                                "minerva",
                                "olympiadBench",
                                ]
    

    for epoch in range(len(saving_path_list)):
        save_path = saving_path_list[epoch]
        
        print(save_path)

        model = Transformer.from_pretrained(model_struct, device=device).train() 
        
        tokenizer = Tokenizer(model_struct+"/tokenizer.json")
        
        dtype_map = {"bfloat16": torch.bfloat16,"float16": torch.float16,"float32": torch.float32}
        
        dtype = dtype_map.get("bfloat16", torch.bfloat16)

        test_data_path_list = train_data_path_list = evalution_data_list
        
        all_result_list = []
        
        for path_number in range(len(evalution_model_weight_list)):
            
            result_list=[]

            if(evalution_model_weight_list[path_number]==""): # "" means the base model
                
                for k in range(len(evalution_data_name_list)):
                    data_name       =   evalution_data_name_list[k]  # 什么数据集
                    train_data_path =   train_data_path_list[k]
                    test_data_path  =   test_data_path_list[k]
                    

                    avg_success, avg_token_length = evaluate(model, 
                                                             tokenizer, 
                                                             device, 
                                                             dtype, 
                                                             train_data_path,
                                                             test_data_path,
                                                             data_name,
                                                             evaluation_batch)

                    print(".................evaluting the base model now.................")
                    print(str(data_name) + " avg-acc: " + str(avg_success) +"\n")
                    print(str(data_name) + " avg-length: " + str(avg_token_length) +"\n")
                    result_list.append([data_name, avg_success, avg_token_length])
            

            else:
                weight_path = evalution_model_weight_list[path_number]
                
                state_dict = torch.load(weight_path, map_location=device)
                
                model.load_state_dict(state_dict, strict=True)
                
                del state_dict
                
                torch.cuda.empty_cache()

                for k in range(len(evalution_data_name_list)):
                    data_name       =   evalution_data_name_list[k]

                    train_data_path =   train_data_path_list[k]

                    test_data_path  =   test_data_path_list[k]
                    
                    avg_success, avg_token_length = evaluate(model, 
                                                             tokenizer, 
                                                             device, 
                                                             dtype, 
                                                             train_data_path,
                                                             test_data_path,
                                                             data_name,
                                                             evaluation_batch)

                    print(".................evaluting the " + evalution_model_weight_list[path_number] + " model now.................")
                    print(str(data_name) + " avg-acc: " + str(avg_success) +"\n")
                    print(str(data_name) + " avg-length: " + str(avg_token_length) +"\n")
                    result_list.append([data_name, avg_success, avg_token_length])
            
            all_result_list.append(result_list)
        
        all_array = np.array(all_result_list)
        
        np.save(save_path,all_array)