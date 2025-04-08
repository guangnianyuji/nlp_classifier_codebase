import pandas as pd
import os
import torch
mode="train"
ti = "20250402_181544"
step = 750000
config='SunyifeiMultiTanghuang48LConfig'

root_path="/mnt/nj-larc/usr/ajie1/sunyifei"

# 定义文件路径模板
file_template = f'{root_path}/nlp_codebase/logs/{mode}/{config}/{ti}/checkpoints/{{}}/result_{step}.csv'

# 初始化一个空的 DataFrame 列表
data_frames = []

# 遍历指定的子目录编号
for i in range(8):  # 假设子目录编号从 0 到 3
    file_path = file_template.format(i)
    if os.path.exists(file_path):
        # 读取 CSV 文件并添加到列表中
        df = pd.read_csv(file_path)
        data_frames.append(df)
    else:
        print(f"文件 {file_path} 不存在，已跳过。")

# 将所有 DataFrame 合并为一个
if data_frames:
    merged_df = pd.concat(data_frames, ignore_index=True)
    # if 'text' in merged_df.columns:
    #     # 删除 'text' 列
    #     merged_df = merged_df.drop(columns=['text'])
    # merged_df['label'] = merged_df['label'].apply(lambda x: eval(x, {"torch": torch, "tensor": torch.tensor}).item())
    # 保存合并后的 DataFrame 到新的 CSV 文件
    print(merged_df.shape)
    output_path = f'{root_path}/nlp_codebase/logs/{mode}/{config}/{ti}/checkpoints/result_{step}_merge.csv'
    merged_df.to_csv(output_path, index=False)
    print(f"合并后的文件已保存到 {output_path}")
else:
    print("未找到任何可合并的 CSV 文件。")
