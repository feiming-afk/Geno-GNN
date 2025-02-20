from torch_geometric.loader import DataLoader
import pandas as pd
from geno_pred import *

def month_min( column, grouped, all_months):
    monthly_data = {month: None for month in all_months}

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

def get_seqs_time_vary(n):
    folder_path = "../../mdl_compare/gisaid_sample_2209_2402"
    results_folder = "results/bianhuan_torch"

    try:
        os.makedirs(results_folder, exist_ok=True)
        print(f"文件夹 '{results_folder}' 已创建或已存在。")
    except OSError as e:
        print(f"创建文件夹 '{results_folder}' 时出错：{e}")

    start_date = pd.Timestamp(year=2020, month=3, day=1)
    end_date = pd.Timestamp(year=2024, month=2, day=1)
    all_months = pd.date_range(start=start_date, end=end_date, freq='MS')

    # 获取已经处理过的文件列表
    processed_files = set(os.listdir(results_folder))
    results = {}

    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename not in processed_files:

                print(f"Processing file: {filename}")
                file_path = os.path.join(root, filename)
                df_raw = pd.read_csv(file_path,low_memory=False)

                dropind = []
                seqs_raw = [seq[0][330:531] for seq in df_raw[['seqs']].values]
                for i in range(len(seqs_raw)):
                    if '*' in seqs_raw[i] or not seqs_raw[i].endswith('T'):
                        dropind.append(i)
                df_drop = df_raw.drop(index=dropind)
                df = df_drop[df_drop['country'].str.strip().str.lower() != 'hong kong']
                print('Hong Kong' in df['country'].unique())
                seqs = [seq[0][330:531] for seq in df[['seqs']].values]

                print('Begin Featurize!!')
                dataset = get_data(seqs,n)
                loader = DataLoader(dataset, batch_size=5000, shuffle=False)

                print('Begin Pred!!')
                df = predcess(df, loader)
                df.to_csv(os.path.join(results_folder, filename), index=False)
            else:
                print(f"Skipping already processed file: {filename}")
                df = pd.read_csv(os.path.join(results_folder, filename),low_memory=False)

            print('Begin Stat!!')
            df['Collection_date'] = pd.to_datetime(df['Collection_date'])
            grouped = df.groupby([df['Collection_date'].dt.year, df['Collection_date'].dt.month])
            columns_to_process = ['affinity','wt','vac','ba1','ba2','ba5']

            for column in columns_to_process:
                if column not in results:
                    results[column] = []
                monthly_df = month_min(column, grouped, all_months)
                results[column].append(monthly_df)

            print('A Loop End!!')
    print('Begin Concat!!')

    tongji_folder = 'results/tongji_torch'
    try:
        os.makedirs(tongji_folder, exist_ok=True)
        print(f"文件夹 '{tongji_folder}' 已创建或已存在。")
    except OSError as e:
        print(f"创建文件夹 '{tongji_folder}' 时出错：{e}")

    for column, dfs in results.items():
        combined_df = pd.concat(dfs, axis=1)
        combined_df.to_csv(os.path.join(tongji_folder, f'combined_{column}.csv'), index=True)

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
        result_df.to_csv(os.path.join(tongji_folder, f'{column}_summary.csv'), index=True)

if __name__ == '__main__':
    # 并行数量
    n = 30
    get_seqs_time_vary(n)