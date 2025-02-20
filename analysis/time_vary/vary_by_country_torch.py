# 根据疫苗分类国家计算fitness结果，还有随机打乱
import pandas as pd
import os
from tqdm import tqdm
import time
import numpy as np
import concurrent.futures

def month_mean(column, grouped, all_months):
    monthly_data = {month: {'mean': np.nan, 'min': np.nan, 'q1': np.nan, 'median': np.nan, 'q3': np.nan, 'max': np.nan} for month in all_months}
    for (year, month), group in grouped:
        mean_values = group[column].mean()
        quantiles = group[column].quantile([0, 0.25, 0.5, 0.75, 1.0])
        date = pd.Timestamp(year=year, month=month, day=1)
        monthly_data[date] = mean_values
        # 保存均值和分位数
        monthly_data[date] = {
            "mean": mean_values,
            "min": quantiles[0],
            "q1": quantiles[0.25],
            "median": quantiles[0.5],
            "q3": quantiles[0.75],
            "max": quantiles[1.0]
        }

    monthly_df = pd.DataFrame.from_dict(monthly_data, orient='index')
    return monthly_df

def month_mean_country(column, grouped, all_months):
    """计算指定列的月平均值、最小值、第一四分位数、中位数、第三四分位数和最大值，处理缺失值"""
    monthly_data = {
        month: {'mean': np.nan, 'min': np.nan, 'q1': np.nan, 'median': np.nan, 'q3': np.nan, 'max': np.nan} for
        month in all_months}
    for (year, month), group in grouped:
        date = pd.Timestamp(year=year, month=month, day=1)
        if not group[column].empty:  # 检查是否有数据
            quantiles = group[column].quantile([0, 0.25, 0.5, 0.75, 1.0])
            monthly_data[date] = {
                "mean": group[column].mean(),
                "min": quantiles[0],
                "q1": quantiles[0.25],
                "median": quantiles[0.5],
                "q3": quantiles[0.75],
                "max": quantiles[1.0]
            }
    return pd.DataFrame.from_dict(monthly_data, orient='index')

def tongji_by_vac(vac_df,save_folder):
    # 根据疫苗种类计算每个种类月均值
    # 替换国家名
    vac_df['country'] = vac_df['country'].replace({'Czechia': 'Czech Republic', 'United States': 'USA'})

    start_date = pd.Timestamp(year=2020, month=3, day=1)
    end_date = pd.Timestamp(year=2024, month=2, day=1)
    all_months = pd.date_range(start=start_date, end=end_date, freq='MS')
    columns_to_process = ['affinity', 'wt', 'vac', 'ba1', 'ba2', 'ba5']
    # 使用字典来存储每个column对应的所有DataFrame
    results = {}
    all_results = {}  # 存储所有国家所有样本的结果
    sample_counts = {}  # 存储每次采样每个国家的记录数量
    for sample in tqdm(range(100)):  # 100次采样
        df = pd.read_csv(f'results/bianhuan_torch/combined_sample_{2022 + sample}.csv',
                         low_memory=False)

        # 根据疫苗筛选国家
        df = df[df['country'].isin(vac_df['country'])]
        df['Collection_date'] = pd.to_datetime(df['Collection_date'])
        # 统计所有
        for column in columns_to_process:
            grouped = df.groupby([df['Collection_date'].dt.year, df['Collection_date'].dt.month])
            monthly_df = month_mean(column, grouped, all_months)
            # 汇总到结果字典
            if column not in results:
                results[column] = []
            results[column].append(monthly_df)
        # 统计国家
        # 统计每次采样的记录数量
        sample_counts[sample] = df.groupby('country').size()
        for country in vac_df['country'].unique():
            country_df = df[df['country'] == country]
            for column in columns_to_process:
                grouped_country = country_df.groupby(
                    [country_df['Collection_date'].dt.year, country_df['Collection_date'].dt.month])
                monthly_df = month_mean_country(column, grouped_country, all_months)
                monthly_df['country'] = country
                monthly_df['column'] = column
                monthly_df = monthly_df.reset_index().rename(columns={'index': 'date'})
                if sample == 0:  # 第一次循环初始化
                    all_results[(column, country)] = monthly_df
                else:
                    all_results[(column, country)] = pd.concat([all_results[(column, country)], monthly_df],
                                                               ignore_index=True)

    # 存储每个国家值
    final_df = pd.concat([all_results[key] for key in all_results.keys()], ignore_index=True)
    final_df.to_csv(os.path.join(save_folder, 'all_country_data.csv'), index=False)
    # 将采样记录数量保存到 CSV 文件
    sample_counts_df = pd.DataFrame.from_dict(sample_counts, orient='index')
    sample_counts_df.to_csv(os.path.join(save_folder, 'sample_counts.csv'))

    # 汇总每个column的所有月均值
    for column, monthly_dfs in results.items():
        combined_df = pd.concat(monthly_dfs, axis=1)
        combined_df.to_csv(os.path.join(save_folder, f'combined_{column}.csv'), index=True)
        # 计算每个列的均值和Boxplot五数的均值
        # 这里我们用列名的基础形式加上列的顺序，以确保唯一性
        row_mean_df = combined_df.filter(like='mean').mean(axis=1).to_frame(name=f'{column}_mean')
        row_min_df = combined_df.filter(like='min').mean(axis=1).to_frame(name=f'{column}_min')
        row_q1_df = combined_df.filter(like='q1').mean(axis=1).to_frame(name=f'{column}_q1')
        row_median_df = combined_df.filter(like='median').mean(axis=1).to_frame(name=f'{column}_median')
        row_q3_df = combined_df.filter(like='q3').mean(axis=1).to_frame(name=f'{column}_q3')
        row_max_df = combined_df.filter(like='max').mean(axis=1).to_frame(name=f'{column}_max')

        # 合并所有汇总数据框
        result_df = pd.concat([row_mean_df, row_min_df, row_q1_df, row_median_df, row_q3_df, row_max_df], axis=1)

        # 保存均值和五数汇总到CSV文件
        result_df.to_csv(os.path.join(save_folder, f'{column}_summary.csv'), index=True)

def trans_format(df):
    # 将日期列转换为日期时间对象
    df['date'] = pd.to_datetime(df['date'])

    # 提取年份和月份
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    # 透视数据
    df_pivot = df.pivot_table(index=['country', 'year', 'month'], columns='column', values=['mean'])

    # 重命名列
    df_pivot.columns = ['_'.join(col).strip() for col in df_pivot.columns.values]

    # 重设索引
    df_pivot = df_pivot.reset_index()

    # 重命名列以匹配你的目标格式
    df_pivot = df_pivot.rename(columns={
        'country': 'country',
        'year': 'year',
        'month': 'month',
        'mean_affinity': 'affinity',
        'mean_wt': 'wt',
        'mean_vac': 'vac',
        'mean_ba1': 'ba1',
        'mean_ba2': 'ba2',
        'mean_ba5': 'ba5'
    })

    return df_pivot

def random_mean(column, grouped, all_months):
    # 计算按月分组的均值
    monthly_data = {month: {'mean': np.nan} for month in all_months}
    for (year, month), group in grouped:
        mean_values = group[column].mean()
        date = pd.Timestamp(year=year, month=month, day=1)
        monthly_data[date]['mean'] = mean_values
    monthly_df = pd.DataFrame.from_dict(monthly_data, orient='index')
    return monthly_df

def random_vac(inac_df, adeno_df, mrna_df, random_state):
    # 随机打乱国家标签检验结果
    # 设置随机种子
    np.random.seed(random_state)

    # 合并所有国家列表
    all_countries = pd.concat([inac_df['country'], adeno_df['country'], mrna_df['country']])

    # 计算每个文件原先的类别比例
    total_countries = len(all_countries)
    prop1 = len(inac_df) / total_countries
    prop2 = len(adeno_df) / total_countries

    # 随机打乱国家
    shuffled_countries = np.random.permutation(all_countries)

    # 根据原比例重新分配国家
    split1 = int(total_countries * prop1)
    split2 = int(total_countries * prop2)

    # 将新的国家列表保存回各自文件中
    inac_df['country'] = shuffled_countries[:split1]
    adeno_df['country'] = shuffled_countries[split1:split1 + split2]
    mrna_df['country'] = shuffled_countries[split1 + split2:]

    return inac_df, adeno_df, mrna_df

def tongji_by_random(vac_df, columns_to_process, all_months):
    # 根据疫苗种类计算每个种类月均值
    # 替换国家名
    vac_df['country'] = vac_df['country'].replace({'Czechia': 'Czech Republic', 'United States': 'USA'})

    # 使用字典来存储每个column对应的所有DataFrame
    results = {column: [] for column in columns_to_process}

    for sample in tqdm(range(100)):  # 假设有100次采样
        ds = pd.read_csv(f'results/bianhuan_torch/combined_sample_{2022 + sample}.csv',
                         low_memory=False)

        # 根据疫苗筛选国家
        ds = ds[ds['country'].isin(vac_df['country'])]
        ds['Collection_date'] = pd.to_datetime(ds['Collection_date'])
        # 统计所有
        for column in columns_to_process:
            grouped = ds.groupby([ds['Collection_date'].dt.year, ds['Collection_date'].dt.month])
            monthly_df = random_mean(column, grouped, all_months)
            results[column].append(monthly_df['mean'])

    # 这里results是一个字典，key是column，value是一个包含100个Series的list
    # 需要将每个list中的Series合并成一个DataFrame，再计算平均值
    final_results = {}
    for column, monthly_means in results.items():
        combined_df = pd.concat(monthly_means, axis=1)
        final_results[column] = combined_df.mean(axis=1)

    return final_results

def process_samples(vac_dfs, columns_to_process, all_months, random_states):
    # 处理多个随机置换并计算结果
    chunk_results_inac = {column: pd.DataFrame(index=all_months) for column in columns_to_process}
    chunk_results_adeno = {column: pd.DataFrame(index=all_months) for column in columns_to_process}
    chunk_results_mrna = {column: pd.DataFrame(index=all_months) for column in columns_to_process}

    for random_state in random_states:
        # 随机置换国家
        inac_df, adeno_df, mrna_df = random_vac(vac_dfs['inac'], vac_dfs['adeno'], vac_dfs['mrna'], random_state)

        results_inac = tongji_by_random(inac_df, columns_to_process, all_months)
        results_adeno = tongji_by_random(adeno_df, columns_to_process, all_months)
        results_mrna = tongji_by_random(mrna_df, columns_to_process, all_months)

        for col in columns_to_process:
            chunk_results_inac[col][f'random_{random_state}'] = results_inac[col]
            chunk_results_adeno[col][f'random_{random_state}'] = results_adeno[col]
            chunk_results_mrna[col][f'random_{random_state}'] = results_mrna[col]

    return chunk_results_inac, chunk_results_adeno, chunk_results_mrna

# 封装函数
def create_folder(path):
    """创建文件夹并处理错误"""
    try:
        os.makedirs(path, exist_ok=True)
        print(f"文件夹 '{path}' 已创建或已存在。")
    except OSError as e:
        print(f"创建文件夹 '{path}' 时出错：{e}")

def save_results(final_data, save_root, prefix):
    """保存结果到指定路径"""
    save_pth = os.path.join(save_root, f'{prefix}_random')
    create_folder(save_pth)

    for col, df in final_data.items():
        if df.empty:
            print(f"Warning: {prefix}[{col}] is empty!")
        else:
            print(f"{prefix}[{col}] contains data, saving...")
            save_pp = os.path.join(save_pth, f'{col}_random_summary.csv')
            df.to_csv(save_pp)

def process_vac_data(vac_df, save_root, prefix):
    """处理单个疫苗数据"""
    save_pth = os.path.join(save_root, prefix)
    create_folder(save_pth)
    tongji_by_vac(vac_df, save_pth)

def process_and_merge_data(save_root, output_path):
    """读取、转换并合并疫苗数据"""
    input_paths = {
        'mrna': os.path.join(save_root, 'mrna/all_country_data.csv'),
        'adeno': os.path.join(save_root, 'adeno/all_country_data.csv'),
        'inac': os.path.join(save_root, 'inac/all_country_data.csv'),
        'mix': os.path.join(save_root, 'mix/all_country_data.csv')
    }

    # 读取并转换格式
    data_frames = []
    for prefix, path in input_paths.items():
        print(f"读取数据文件：{path}")
        df = pd.read_csv(path)
        transformed_df = trans_format(df)
        data_frames.append(transformed_df)

    # 合并数据
    merged_df = pd.concat(data_frames)
    merged_df.to_csv(output_path, index=False)
    print(f"所有数据已合并并保存到：{output_path}")

if __name__ == '__main__':
    # 文件路径
    vac_paths = {
        'inac': 'data/vac_country/country_inactivated_vaccine_low_2110.csv',
        'adeno': 'data/vac_country/country_adenovirus_vector_low_2110.csv',
        'mrna': 'data/vac_country/country_mRNA_low_2110.csv',
        'mix': 'data/vac_country/country_low_mix_2110.csv'
    }
    save_root = 'results/vac_tongji'
    create_folder(save_root)

    # 读取数据
    vac_dfs = {prefix: pd.read_csv(path, low_memory=False) for prefix, path in vac_paths.items()}
    # 统计各疫苗数据
    for prefix, df in vac_dfs.items():
        process_vac_data(df, save_root, prefix)

    # 读取、转换并合并数据
    output_path = os.path.join(save_root, 'monthly_country.csv')
    process_and_merge_data(save_root, output_path)


    # 文件路径
    vac_paths = {
        'inac': 'data/vac_country/country_inactivated_vaccine_low_2110.csv',
        'adeno': 'data/vac_country/country_adenovirus_vector_low_2110.csv',
        'mrna': 'data/vac_country/country_mRNA_low_2110.csv'
    }
    # 读取数据
    vac_dfs = {prefix: pd.read_csv(path, low_memory=False) for prefix, path in vac_paths.items()}
    # 时间和列配置
    start_date = pd.Timestamp(year=2020, month=3, day=1)
    end_date = pd.Timestamp(year=2024, month=2, day=1)
    all_months = pd.date_range(start=start_date, end=end_date, freq='MS')
    columns_to_process = ['affinity', 'wt', 'vac', 'ba1', 'ba2', 'ba5']
    # 初始化最终数据结构
    final_data = {prefix: {col: pd.DataFrame(index=all_months) for col in columns_to_process} for prefix in
                  vac_dfs.keys()}
    # 并行处理
    num_workers = 10
    random_states = list(range(100))
    chunk_size = 10
    chunks = [random_states[i:i + chunk_size] for i in range(0, len(random_states), chunk_size)]
    print(vac_dfs.keys())
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_samples, vac_dfs, columns_to_process, all_months, chunk)
            for chunk in chunks
        ]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            chunk_results = future.result()
            for prefix, results in zip(vac_dfs.keys(), chunk_results):
                for col in final_data[prefix]:
                    final_data[prefix][col] = pd.concat([final_data[prefix][col], results[col]], axis=1)

    # 保存结果
    for prefix in vac_dfs.keys():
        save_results(final_data[prefix], save_root, prefix)





