import pandas as pd
import os
import itertools

# 定义路径和文件夹
data_path = r'results/vac_tongji'
folders = ['adeno_random', 'inac_random', 'mrna_random']
files = ['affinity_random_summary.csv', 'vac_random_summary.csv', 'wt_random_summary.csv']

# 初始化空列表来存储结果
results = []

# 生成文件夹的组合
folder_combinations = itertools.combinations(folders, 2)

for folder_pair in folder_combinations:
    print(folder_pair)
    vac1 = folder_pair[0]
    vac2 = folder_pair[1]

    dfs = []
    for file in files:
        fitness_name = file.split('_')[0]

        path1 = os.path.join(data_path, vac1, file)
        path2 = os.path.join(data_path, vac2, file)

        df1 = pd.read_csv(path1)
        df2 = pd.read_csv(path2)

        new_dfs = []
        for i in range(100):
            new_df1 = pd.DataFrame({ fitness_name: ['NA']*48, 'sequence': range(48), 'vac_type':[vac1]*48, 'dum':[1]*48 })
            new_df1['dumseq'] = new_df1['sequence']*new_df1['dum']
            new_df1['random_state'] = i

            new_df2 = pd.DataFrame(
                {fitness_name: ['NA'] * 48, 'sequence': range(48), 'vac_type': [vac2] * 48, 'dum': [0] * 48})
            new_df2['dumseq'] = new_df2['sequence'] * new_df2['dum']
            new_df2['random_state'] = i

            col_name = f'random_{i}'
            new_df1[fitness_name] = df1[col_name]
            new_df2[fitness_name] = df2[col_name]

            new_df = pd.concat([new_df1, new_df2])
            new_dfs.append(new_df)

        df = pd.concat(new_dfs)
        dfs.append(df)
    print(dfs[0])
    print(dfs[1])
    print(dfs[2])
    final_df = dfs[0].merge(dfs[1]).merge(dfs[2])
    save_path = os.path.join(data_path, f'jianyan_{vac1}_{vac2}.csv')
    final_df.to_csv(save_path, index=False)



