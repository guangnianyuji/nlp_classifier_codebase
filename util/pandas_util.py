import pandas as pd
def repeat_data(a):
        # 复制 label=1 的数据 10 倍
    # 复制 label=1 的数据 10 倍
    label_1_data = a[a['label'] == 1]
    label_1_data_repeated = pd.concat([label_1_data] * 10, ignore_index=True)

    # 选择 120000 行 label=0 的数据
    label_0_data = a[a['label'] == 0]#.sample(n=120000, random_state=42, replace=True)

    # 合并数据
    enhanced_df = pd.concat([label_1_data_repeated, label_0_data])

    return enhanced_df




def round_data(df,batch_size):
    current_size = len(df)
    remainder = current_size % batch_size
    if remainder != 0:
        # 计算需要补充的样本数量
        samples_to_add = batch_size - remainder

        # 从现有数据集中随机选择需要补充的样本
        additional_samples = df.sample(n=samples_to_add, replace=True, random_state=42)

        # 将补充的样本添加到数据集中
        df = pd.concat([df, additional_samples])    
    return df

