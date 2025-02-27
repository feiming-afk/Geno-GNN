import pandas as pd
import re
import numpy as np

df = pd.read_csv('results/mut_or_not_all_mp.csv', low_memory=False)

# 提取列名中的三位数字
cols_to_process = df.columns[9:]  # 从第五列开始
num_map = {}
for col in cols_to_process:
    num = col[1:4]
    if num not in num_map:
        num_map[num] = []
    num_map[num].append(col)

print(f"列名映射: {num_map}")
print('向量化数据处理!')
# 向量化数据处理
for num, cols in num_map.items():
    if len(cols) == 2:
        # 使用向量化操作
        df_subset = df[cols]
        mask = (df_subset[cols[0]] == 'T') & (df_subset[cols[1]] == 'F')
        df.loc[mask, cols[1]] = np.nan
        mask = (df_subset[cols[0]] == 'F') & (df_subset[cols[1]] == 'T')
        df.loc[mask, cols[0]] = np.nan
    if len(cols) == 3:
        # 使用向量化操作
        df_subset = df[cols]
        mask = (df_subset[cols[0]] == 'T') & (df_subset[cols[1]] == 'F') & (df_subset[cols[2]] == 'F')
        df.loc[mask, cols[1]] = np.nan
        df.loc[mask, cols[2]] = np.nan
        mask = (df_subset[cols[0]] == 'F') & (df_subset[cols[1]] == 'T') & (df_subset[cols[2]] == 'F')
        df.loc[mask, cols[0]] = np.nan
        df.loc[mask, cols[2]] = np.nan
        mask = (df_subset[cols[0]] == 'F') & (df_subset[cols[1]] == 'F') & (df_subset[cols[2]] == 'T')
        df.loc[mask, cols[0]] = np.nan
        df.loc[mask, cols[1]] = np.nan

for num, cols in num_map.items():
    if len(cols) > 1:
        df_filtered = df[(df[cols[0]] == 'T') & (df[cols[1]] == 'T')]
        print(f"三位数为 {num}，两列都为 'T' 的行：\n{df_filtered}\n")

print(df) #查看修改后的df
print(df.isnull().sum()) # 检查空值数量
df.to_csv('results/mut_or_not_all_mp_nan.csv',index=False)

def stat_tf_by_01(df):
    # 定义需要计算比例的列
    cols_to_calculate = df.columns[4:9]  # 第五到第九列
    tf_cols = df.columns[9:]  # 第十列到最后一列

    # 存储结果的列表
    results = []

    for col in cols_to_calculate:
    # 遍历免疫背景

        # 计算贡献
        # 计算1和0的个数
        count_1 = len(df[df[col] == 1]) # 逃逸数量
        count_0 = len(df[df[col] == 0]) # 不逃逸数量

        # 计算比例
        total_10 = count_1 + count_0 # 总数
        flag_1 = (count_1 > (total_10 * 0.05)) # 逃逸与总数5%
        flag_0 = (count_0 > (total_10 * 0.05)) # 不逃逸与总数5%
        proportion_1 = count_1 / total_10 if total_10 > 0 else 0 # 逃逸占比
        proportion_0 = count_0 / total_10 if total_10 > 0 else 0 # 不逃逸占比

        for value in [0, 1]:
            # 筛选出当前列值为value的行
            df_subset = df[df[col] == value]

            proportion_ib = np.nan
            if value == 0:
                proportion_ib = proportion_0
            if value == 1:
                proportion_ib = proportion_1

            # 循环处理突变
            for tf_col in tf_cols:

                # 计算affinity情况
                median_t = np.median(df.loc[df[tf_col] == 'T', 'affinity'])
                median_f = np.median(df.loc[df[tf_col] == 'F', 'affinity'])

                diff_tf = median_t - median_f
                diff_ft = median_f - median_t

                # 筛选出当前列值为'T'或'F'的行，忽略空值
                df_tf_subset = df_subset[df_subset[tf_col].isin(['T', 'F'])]

                # 计算T和F的个数
                count_t = len(df_tf_subset[df_tf_subset[tf_col] == 'T']) # 有这个突变的数量
                count_f = len(df_tf_subset[df_tf_subset[tf_col] == 'F']) # 没这个突变的数量

                # 计算比例
                total_tf = count_t + count_f
                proportion_t = count_t / total_tf if total_tf > 0 else 0
                proportion_f = count_f / total_tf if total_tf > 0 else 0

                binary_t = 0
                binary_f = 0
                if proportion_t > 0.5 and flag_1 == True: # 逃逸的数量里有这个突变的比例大于0.5，逃逸总数大于总数0.05
                    binary_t = 1
                if proportion_f > 0.5 and flag_0 == True:
                    binary_f = 1
                contribution_t = proportion_t * proportion_ib * binary_t
                contribution_f = proportion_f * proportion_ib * binary_f
                print(col, value, tf_col, count_t, count_f, median_t, median_f, proportion_t, proportion_f)

                # 将结果添加到列表中
                results.append([tf_col, 'T', col, value, median_t, diff_tf, proportion_t, binary_t, proportion_ib, contribution_t])
                results.append([tf_col, 'F', col, value, median_f, diff_ft, proportion_f, binary_f, proportion_ib, contribution_f])

    # 创建第一个DataFrame
    result_df = pd.DataFrame(results, columns=['mutations', 'TF_Value', 'immune', 'Value', 'affinity', 'affinity_diff', 'Proportion', 'Binary', 'Prop_IB', 'Contribution'])
    print("第一个DataFrame:\n", result_df)
    return result_df

# 加载数据
result_df = stat_tf_by_01(df)
result_df.to_csv('results/result_tf_mp.csv', index=False)
result_df2 = result_df[(result_df['TF_Value'] == 'T') & (result_df['Value'] == 1)]
result_df2.to_csv('results/result_tf_mp_t.csv', index=False)


df = pd.read_csv('results/mut_or_not_convergent_comp.csv', low_memory = False)
result_df = stat_tf_by_01(df)
result_df.to_csv('results/result_tf_convergent.csv', index=False)
result_df2 = result_df[(result_df['TF_Value'] == 'T') & (result_df['Value'] == 1)]
result_df2.to_csv('results/result_tf_convergent_t.csv', index=False)