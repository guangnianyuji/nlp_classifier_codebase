import pandas as pd
import numpy as np
import torch

# Load the data
df = pd.read_csv('/mnt/nj-larc/usr/ajie1/sunyifei/nlp_codebase/logs/train/SunyifeiBertConfig/20250317_164900/checkpoints/result_10000_merge_91864.csv')

# 使用 eval 时明确传递 tensor 作为 torch.tensor
df['label'] = df['label'].apply(lambda x: eval(x, {"torch": torch, "tensor": torch.tensor}).item())

# Convert to list
labels = df['label'].to_list()
# print()
# For demonstration purposes, we can ignore the comments part for now
score_border = np.arange(0, 1, 0.01)
score_list = df['score_1'].to_list()

for border in score_border:
    predict = [1 if score > border else 0 for score in score_list]
    TP = FP = FN = TN = 0
    TP = sum([1 for i in range(len(labels)) if labels[i] == 1 and predict[i] == 1])
    FP = sum([1 for i in range(len(labels)) if labels[i] == 0 and predict[i] == 1])
    FN = sum([1 for i in range(len(labels)) if labels[i] == 1 and predict[i] == 0])
    TN = sum([1 for i in range(len(labels)) if labels[i] == 0 and predict[i] == 0])
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    print('border:', border)
    print("precision:",precision)
    print("recall:", recall)
    print("-----------------")