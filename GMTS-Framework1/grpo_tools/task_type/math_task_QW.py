# the dataset preparing for the math task here, use the QW-format for saying.
import re
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Optional
from torch.utils.data import Dataset
from grpo_tools.data_types import MATH_MiniBatch
from grpo_tools.tokenizer import Tokenizer
from grpo_tools.utils.parser import *
from grpo_tools.utils.math_normalization import *
from grpo_tools.utils.grader import *

# Use QW evaluation message.
SYSTEM_MESSAGE = ( "Please reason step by step, and put your final answer within \\boxed{}." )

class mathTasksDataset(Dataset):
    """Prepare MATH Tasks for training and testing """
    def __init__(
        self,
        tokenizer: Tokenizer,
        train_data_path: str,
        test_data_path: str,
        split: str = "train",
        data_name: str = 'math',
    ):  
        
        if(data_name=='math'): # math
            train_data = pd.read_json(Path(train_data_path) /  "train_data/train.jsonl", lines=True)
            test_data  = pd.read_json(Path(test_data_path)  /  "test_data/test.jsonl", lines=True)
        
        elif ((data_name=='aime')):
            train_data = pd.read_csv(Path(train_data_path)/ "AIME_Dataset.csv")
            test_data =  pd.read_csv(Path(test_data_path)/ "AIME_Dataset.csv")
        
        elif ((data_name=='amc')):
            train_data = pd.read_parquet(Path(train_data_path) / "data")
            test_data  = pd.read_parquet(Path(test_data_path) / "data")
        
        elif(data_name=='olympiadBench'):
            train_data = pd.read_json(Path(train_data_path) /  "OlympiadBench.jsonl", lines=True)
            test_data =  pd.read_json(Path(test_data_path) /   "OlympiadBench.jsonl", lines=True)
        
        elif(data_name=='minerva'):
            train_data = pd.read_json(Path(train_data_path) / "test/minerva.json")
            test_data =  pd.read_json(Path(train_data_path) / "test/minerva.json")
        
        self.data = (train_data if split == "train" else test_data) # train or test_data
        
        # change the column name.
        if(data_name=='math'): 
            self.data = self.data.rename(columns={"problem": "question"})
        
        elif((data_name=='aime')):
            self.data = self.data.rename(columns={"Question": "question"})
            self.data = self.data.rename(columns={"Answer": "answer"})
        
        elif(data_name=='olympiadBench'):
            self.data = self.data.rename(columns={"final_answer": "answer"})
        
        elif(data_name=='minerva'):
            self.data = self.data.rename(columns={"problem": "question"})
        
        self.tokenizer = tokenizer


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx].to_dict()
        item.update(self.encode_prefix(item["answer"], item["question"]))
        return item
    
    def encode_prefix(self, answer: str, question: str):
        prefix = self.tokenizer.encode_chat_with_response_prompt(
            [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": question},
            ],
            "",
        )

        tokens = self.tokenizer.tokenize(prefix)

        return {
            "prefix": prefix,                  # model-input-sentence
            "prefix_tokens": tokens.tokens,    # model-input-token
            "prefix_token_ids": tokens.ids,    # model-input-token-ids
        }
    
    
    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> MATH_MiniBatch:
        """Collate examples into a batch."""

        answers = [item["answer"] for item in batch]
        questions = [item["question"] for item in batch]
        prefix = [item["prefix"] for item in batch]
        prefix_tokens = [item["prefix_tokens"] for item in batch]
        prefix_token_ids = [item["prefix_token_ids"] for item in batch]
        
        return MATH_MiniBatch(
            questions=questions,
            answers=answers,
            prefix=prefix,
            prefix_tokens=prefix_tokens,
            prefix_token_ids=prefix_token_ids,
        )


def answer_reward_function(response: str, 
                           answers: str = '',
                           data_name: str = 'gsm8k' ) -> float:
    
    """
    Checks if the answer uses all numbers exactly once and evaluates to the target
    """
    answer_content = extract_answer(response, "math") # get the answer here.

    if(data_name=='math'):
        ground_truth = str(answers) 
    
    elif((data_name=='aime')):
        ground_truth = str(answers) 
    
    elif ((data_name=='amc')):
        ground_truth = str(answers) 
    
    elif(data_name=='olympiadBench'):
        ground_truth = str(answers[0])
    
    elif(data_name=='minerva'):
        ground_truth = str(answers[0])
    
    is_correct = check_is_correct(answer_content, ground_truth) # check is correct?

    if(is_correct):
        return 1.0
    
    return 0.0


def math_reward_function(
    response: str,
    answers: str,
    end_token: str = None,
    data_name: str = 'gsm8k'
) -> Dict[str, Any]:
    
    answer_reward = answer_reward_function(response, answers, data_name)
    return {
        "reward": answer_reward,
        "reward_info": {
            "format_reward": 0.0,
            "answer_reward": answer_reward,
        },
    }