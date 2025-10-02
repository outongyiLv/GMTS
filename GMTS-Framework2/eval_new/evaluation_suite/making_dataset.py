from datasets import load_from_disk
from datasets import Dataset
path = "/rgzn/home/zyw/loty/loty-main/eval/evaluation_suite/aime25" # train：表示上述训练集在本地的路径
dataset = load_from_disk(path)