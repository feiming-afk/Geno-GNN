from torch_geometric.loader import DataLoader
import pandas as pd
from geno_pred import *
from itertools import product
import json
import time
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description='MutationSpace')
parser.add_argument('--sig', type=str, default='all')
args = parser.parse_args()

def measure_runtime(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        runtime = end_time - start_time
        hours, rem = divmod(runtime, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"运行时间: {int(hours)}小时 {int(minutes)}分钟 {seconds:.2f}秒")
        return result
    return wrapper

@measure_runtime
def generate_combination(updated_mutants,matched_mutants,mapping_dict_json_path,combinations_file_path):
    # 提取数字并创建映射，同时考虑匹配的变异
    print(updated_mutants)
    print('开始创建映射！')
    updated_number_to_mutant = {}
    for mutant in updated_mutants:
        number = int(''.join(filter(str.isdigit, mutant)))
        is_matched = any(mutant in pair for pair in matched_mutants)
        if not is_matched:
            updated_number_to_mutant.setdefault(number, []).append(mutant)

    print(updated_number_to_mutant)
    print(matched_mutants)
    # 为匹配的变异创建单独的条目
    print('处理匹配变异！')
    for pair in matched_mutants:
        # 使用匹配变异的第一个元素的数字作为键
        pair_number = int(''.join(filter(str.isdigit, pair[0])))
        updated_number_to_mutant[pair_number] = list(pair)
    print(updated_number_to_mutant)
    # 按数字大小排序
    print('排序！')
    updated_ordered_numbers_sorted = sorted(updated_number_to_mutant.keys())
    print(updated_ordered_numbers_sorted)
    # 使用product生成所有可能的组合（按排序后的顺序）
    print('生成可能组合！')
    updated_ordered_combinations_sorted = list(
        product(*[list(range(len(updated_number_to_mutant[number]) + 1)) for number in updated_ordered_numbers_sorted]))
    print(len(updated_ordered_combinations_sorted))
    # 过滤出只在正确位置上有变异的组合
    print('开始过滤！')
    updated_valid_ordered_combinations_sorted = []
    for comb in updated_ordered_combinations_sorted:
        is_valid = True
        for i, state in enumerate(comb):
            if state != 0:
                expected_number = updated_ordered_numbers_sorted[i]
                actual_number = int(''.join(filter(str.isdigit, updated_number_to_mutant[expected_number][state - 1])))
                if expected_number != actual_number:
                    # print(expected_number, actual_number)
                    is_valid = False
                    break
        if is_valid:
            updated_valid_ordered_combinations_sorted.append(comb)

    # 计算总数并显示所有组合
    total_updated_ordered_combinations_sorted = len(updated_valid_ordered_combinations_sorted)
    print('计数:{}'.format(total_updated_ordered_combinations_sorted))
    print(updated_number_to_mutant)
    # 将映射字典保存为JSON文件
    print('保存！')
    with open(mapping_dict_json_path, 'w') as json_file:
        json.dump(updated_number_to_mutant, json_file)

    # 创建组合情况的字符串，并保存为CSV文件
    combination_strings = [''.join(map(str, comb)) for comb in updated_valid_ordered_combinations_sorted]
    combinations_df = pd.DataFrame(combination_strings, columns=['Combination'])
    combinations_df.to_csv(combinations_file_path, index=False)

# 函数：根据组合字符串和映射字典还原变异情况
def restore_mutations(combination_str, mapping_dict):
    # 按照数字大小对键进行排序
    sorted_keys = sorted(mapping_dict.keys(), key=lambda x: int(x))

    mutation_combination = []
    for i, state in enumerate(combination_str):
        number = sorted_keys[i]
        if int(state) > 0:
            if number == '373':
                mutation_combination += mapping_dict[number]
            else:
                mutation_combination.append(mapping_dict[number][int(state) - 1])
    return mutation_combination

@measure_runtime
def generate_seqs(combinations,json_file_path,wt_seq_path):
    # 读取JSON文件
    with open(json_file_path, 'r') as file:
        mapping_dict_from_json = json.load(file)
    # 读取wildtype序列
    with open(wt_seq_path) as g:
        g.readline()
        wt = [i for i in ''.join(g.read().split('\n'))[330:531]]
    # 遍历组合，生成序列
    seqs = []
    for combination in tqdm(combinations):
        temp = wt[:]
        mutation_combination = restore_mutations(combination,mapping_dict_from_json)
        for mutation in mutation_combination:
            pos = int(mutation[1:4]) - 331
            mut = mutation[-1]
            temp[pos] = mut
        seqs.append(''.join(temp))

    return seqs

def chunk_data(df, num_chunks):
    chunk_size = len(df) // num_chunks
    remainder = len(df) % num_chunks
    return [df.iloc[i * chunk_size + min(i, remainder):(i + 1) * chunk_size + min(i + 1, remainder)] for i in range(num_chunks)]

@measure_runtime
def process_data(json_file_path,combinations_file_path,background_path,wt_seq_path,n):
    combinations_df = pd.read_csv(combinations_file_path, low_memory=False, dtype={'Combination': str})
    print(combinations_df.shape)
    print('开始生成序列！')
    seqs = generate_seqs(combinations_df['Combination'],json_file_path,wt_seq_path)
    print('开始特征化！')
    dataset = get_data(seqs,n)
    loader = DataLoader(dataset, batch_size=5000, shuffle=False)
    print('开始预测表型值！')
    complete_df = predcess(combinations_df, loader)
    complete_df.to_csv(background_path, index=False)

@measure_runtime
def multirun():
    # 更新的变异列表，其中某些变异被匹配到一起
    updated_mutantses = [[
        'G339D','G339H','R346T','L368I','S371L','S371F','T376A','D405N','R408S','K444T',
        'V445P','G446S','L452R','F456L','N460K','F486V','F486S','F486P','F490S','Q493R','G496S',
        'S373P','S375F','K417N','N440K','S477N','T478K','E484A','Q498R','N501Y','Y505H'
    ]]

    # 定义匹配的变异对
    matched_mutantses = [
                         [('S373P','S375F','K417N','N440K','S477N','T478K','E484A','Q498R','N501Y','Y505H')]
                         ]

    json_file_paths = ['mediate_results/mapping_dict_all_mp.json']
    combinations_file_paths = ['mediate_results/combinations_all_mp.csv']
    background_paths = ['mediate_results/mutant_background_all_mp.csv']
    wt_seq_path = 'data/B'
    n = 30
    i = 0
    generate_combination(updated_mutantses[i],matched_mutantses[i],json_file_paths[i],combinations_file_paths[i])
    process_data(json_file_paths[i],combinations_file_paths[i],background_paths[i],wt_seq_path,n)

@measure_runtime
def multirun_ba12():
    # 更新的变异列表，其中某些变异被匹配到一起
    updated_mutantses = [['G339D','S371L','S373P','S375F','K417N','N440K','G446S','S477N','T478K','E484A','Q493R','G496S','Q498R','N501Y','Y505H'],
                       ['G339D','S371F','S373P','S375F','T376A','D405N','R408S','K417N','N440K','S477N','T478K','E484A','Q493R','Q498R','N501Y','Y505H']
                       ]

    # 定义匹配的变异对
    matched_mutantses = [[], []]

    json_file_paths = ['mediate_results/mapping_dict_ba1_comp.json','mediate_results/mapping_dict_ba2_comp.json']
    combinations_file_paths = ['mediate_results/combinations_ba1_comp.csv','mediate_results/combinations_ba2_comp.csv']
    background_paths = ['mediate_results/mutant_background_ba1_comp.csv','mediate_results/mutant_background_ba2_comp.csv']
    wt_seq_path = 'data/B'
    n = 30
    for i in range(2):
        generate_combination(updated_mutantses[i],matched_mutantses[i],json_file_paths[i],combinations_file_paths[i])
        process_data(json_file_paths[i],combinations_file_paths[i],background_paths[i],wt_seq_path,n)

@measure_runtime
def multirun_convergent():
    # 更新的变异列表，其中某些变异被匹配到一起
    updated_mutantses = [['S373P','S375F','K417N','N440K','S477N','T478K','E484A','Q498R','N501Y','Y505H']
                       ]

    # 定义匹配的变异对
    matched_mutantses = [[]]

    json_file_paths = ['mediate_results/mapping_dict_convergent_comp.json']
    combinations_file_paths = ['mediate_results/combinations_convergent_comp.csv']
    background_paths = ['mediate_results/mutant_background_convergent_comp.csv']
    wt_seq_path = 'data/B'
    n = 30
    i = 0
    generate_combination(updated_mutantses[i],matched_mutantses[i],json_file_paths[i],combinations_file_paths[i])
    process_data(json_file_paths[i],combinations_file_paths[i],background_paths[i],wt_seq_path,n)

def mutnum_change(df):
    # 计算突变数量
    def count_non_zeros(number):
        # 直接计算非0数字的数量
        return sum(int(digit) != 0 for digit in str(number))

    df['count'] = df['Combination'].apply(count_non_zeros)

    return df

def post_process(df,output_pth,filtered_pth):
    # 计算突变数量
    df = mutnum_change(df)
    columns_to_count = ['wt','vac','ba1','ba2','ba5']
    # Create a new DataFrame with binary values and add it to the original DataFrame
    binary_df = df[columns_to_count].applymap(lambda x: 1 if x > 0.5 else 0)
    binary_df.columns = [col + "_binary" for col in columns_to_count] #Rename columns
    df = pd.concat([df, binary_df], axis=1)
    df['count_greater_than_0.5'] = df[columns_to_count].gt(0.5).sum(axis=1)
    df.to_csv(output_pth, index=0)
    ds = df[['count','count_greater_than_0.5','affinity','wt_binary','vac_binary','ba1_binary','ba2_binary','ba5_binary']]
    ds.to_csv(filtered_pth,index=0)

if __name__ == '__main__':
    sig = args.sig
    if sig == 'all':
        multirun()
        df = pd.read_csv('mediate_results/mutant_background_all_mp.csv', low_memory=False)
        pth = 'results/mut_bk_plot_torch_all_mp.csv'
        pth_filter = 'results/mut_bk_plot_torch_all_mp_filter.csv'
        post_process(df, pth, pth_filter)
    elif sig == 'ba12':
        multirun_ba12()
        df1 = pd.read_csv('mediate_results/mutant_background_ba1_comp.csv', low_memory=False)
        df2 = pd.read_csv('mediate_results/mutant_background_ba2_comp.csv', low_memory=False)
        pth1 = 'results/mut_bk_plot_torch_ba1_comp.csv'
        pth1_filter = 'results/mut_bk_plot_torch_ba1_comp_filter.csv'
        pth2 = 'results/mut_bk_plot_torch_ba2_comp.csv'
        pth2_filter = 'results/mut_bk_plot_torch_ba2_comp_filter.csv'
        post_process(df1,pth1,pth1_filter)
        post_process(df2, pth2, pth2_filter)
    elif sig == 'convergent':
        multirun_convergent()
        df = pd.read_csv('mediate_results/mutant_background_convergent_comp.csv', low_memory=False)
        pth = 'results/mut_bk_plot_torch_convergent_comp.csv'
        pth_filter = 'results/mut_bk_plot_torch_convergent_comp_filter.csv'
        post_process(df, pth, pth_filter)