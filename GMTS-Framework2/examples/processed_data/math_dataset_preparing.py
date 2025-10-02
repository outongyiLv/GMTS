# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the MATH-lighteval dataset to parquet format
"""

import argparse
import json
import os

import datasets
import pandas as pd

from pathlib import Path
from verl.utils.hdfs_io import copy, makedirs
from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == "__main__":

    # data_source = "/rgzn/home/zyw/test_data/MATH-lighteval/data"
    # print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    # dataset = datasets.load_dataset(data_source)
    # train_dataset = dataset["train"]
    # test_dataset = dataset["test"]

    local_dir = "/rgzn/home/zyw/loty/loty-main/data/process_data/math"
    train_data_path = "/rgzn/home/zyw/loty/loty-main/data/download_data/DAPO-Math-17k/train-00000-of-00001.parquet"
    dataset = datasets.load_dataset("parquet", data_files={"train": train_data_path, "test": train_data_path})

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    instruction_following = "Let's think step by step and output the final answer within \\boxed{}."

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("prompt") # question

            question = question + " " + instruction_following

            answer = example.pop("solution")
            solution = answer # 提取的这个就是solution

            data = {
                "data_source": "math",
                "prompt": [{"role": "user", "content": question}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {"split": split, "index": idx},
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = os.path.expanduser(local_dir)


    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
    # Save one example as JSON for reference
    example = train_dataset[0]
    with open(os.path.join(local_dir, "train_example.json"), "w") as f:
        json.dump(example, f, indent=2)
    example = test_dataset[0]
    with open(os.path.join(local_dir, "test_example.json"), "w") as f:
        json.dump(example, f, indent=2)
