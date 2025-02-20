import pandas as pd
import json
import matplotlib.pyplot as plt
import re

def generate_combination_string(mutations, mapping_dict):
    # 根据变异情况和映射字典生成对应的组合数字串
    combination_str = ''
    for number, variants in sorted(mapping_dict.items(), key=lambda x: x[0]):
        # 默认状态为0（无变异）
        state = 0
        for i, variant in enumerate(variants, start=1):
            if variant in mutations:
                if number == '373':
                    state = 1
                    break
                else:
                    state = i  # 如果变异存在，更新状态为对应的索引值
                    break
        combination_str += str(state)
    return combination_str

def generate_clade_combination():
    # 生成clade的组合数字串
    df = pd.read_csv('data/mut_count.csv')
    df = df[(df['clade'] != 'Gamma') & (df['clade'] != 'Beta')
            & (df['clade'] != 'Alpha') & (df['clade'] != 'Delta')]
    # 读取JSON文件
    json_file_path = 'mediate_results/mapping_dict_all_mp.json'
    with open(json_file_path, 'r') as file:
        mapping_dict_from_json = json.load(file)
    print(mapping_dict_from_json)
    clade_comb = {}
    for index,row in df.iterrows():
        muts = row['mutant'].split(';')
        comb_str = generate_combination_string(muts,mapping_dict_from_json)
        clade_comb[row['clade']] = int(comb_str)
    return clade_comb

def clade_select(df):
    # 从df中筛选出需要的clade
    rows = []
    clade_comb = generate_clade_combination()
    #print(clade_comb)
    for key, value in clade_comb.items():
        # 筛选行并转化成list
        select_list = df[df['Combination']==value][['count','affinity','count_greater_than_0.5']]
        #print(select_list)
        select_list = select_list.iloc[0].tolist()
        select_list.append(key)
        rows.append(select_list)
    columns = ['count','affinity','count_greater_than_0.5','clade']
    clade_df = pd.DataFrame(rows,columns=columns)
    return clade_df

def find_non_zero_positions_and_values(s):
    # 查找非零数字的位置和对应的值
    return [(m.start(), m.group()) for m in re.finditer(r'[1-9]', s)]

def mutant_select_all(df, json_file_path):
    df = df[['Combination', 'count', 'affinity', 'count_greater_than_0.5','wt_binary','vac_binary','ba1_binary','ba2_binary','ba5_binary']].copy()
    with open(json_file_path, 'r') as file:
        mapping_dict_from_json = json.load(file)
    for pos, muts in mapping_dict_from_json.items():
        # print(pos)
        # print(muts)
        for mut in muts:
            aa = generate_combination_string(mut, mapping_dict_from_json)
            positions, values = find_non_zero_positions_and_values(aa)[0]
            n = len(aa) - positions
            value = int(values)
            df.loc[:, mut] = df['Combination'].apply(lambda x: 'T' if (int(x) // 10 ** (n - 1)) % 10 == value else 'F')
    return df

if __name__ == '__main__':

    df = pd.read_csv('results/mut_bk_plot_torch_all_mp.csv',low_memory=False)
    print(df.columns)
    dd = clade_select(df)
    dd.to_csv('results/seleted_rows_mp.csv',index = False)

    json_file_path = 'mediate_results/mapping_dict_all_mp.json'
    dddf = mutant_select_all(df, json_file_path)
    dddf.to_csv('results/mut_or_not_all_mp.csv', index = False)

    df = pd.read_csv('results/mut_bk_plot_torch_convergent_comp.csv',low_memory=False)
    json_file_path = 'mediate_results/mapping_dict_convergent_comp.json'
    dddf = mutant_select_all(df, json_file_path)
    dddf.to_csv('results/mut_or_not_convergent_comp.csv', index = False)





