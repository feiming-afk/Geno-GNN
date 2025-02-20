import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 读取 CSV 文件
data = pd.read_csv('results/vac_tongji/monthly_country.csv')

# 筛选掉不用的国家
data = data[(data['country'] != 'Slovakia') & (data['country'] != 'Puerto Rico') & (data['country'] != 'Bahrain')]
# 读取其他三个CSV文件
csv1 = pd.read_csv('data/vac_country/country_inactivated_vaccine_low_2110.csv')
csv2 = pd.read_csv('data/vac_country/country_adenovirus_vector_low_2110.csv')
csv3 = pd.read_csv('data/vac_country/country_mRNA_low_2110.csv')
csv4 = pd.read_csv('data/vac_country/country_low_mix_2110.csv')


data = data.replace({'Czech Republic':'Czechia','USA':'United States'})
# 定义函数为每个国家分类
categories = {}
for index,row in data.iterrows():
    country = row['country']
    if country in csv1['country'].values:
        categories[country] = 'Inactivated vaccinees'
    elif country in csv2['country'].values:
        categories[country] = 'Adenovirus vaccinees'
    elif country in csv3['country'].values:
        categories[country] = 'mRNA vaccinees'
    elif country in csv4['country'].values:
        categories[country] = 'Mix vaccinees'

print(categories)

# 新增 year-month 列
data['year-month'] = pd.to_datetime(data['year'].astype(str) + '-' + data['month'].astype(str))

# 筛选出所有早于 2022 年 9 月的记录
data = data[data['year-month'] < '2022-02-01']
data = data[data['year-month'] > '2020-11-01']
print(data.columns[3:9])
# 初始化 MinMaxScaler
scaler = MinMaxScaler()  # 或者使用 StandardScaler()

# 对 3 到 8 列进行归一化，假设列索引从 0 开始
data.iloc[:, 3:9] = scaler.fit_transform(data.iloc[:, 3:9])
df = data.drop(columns=['year','month'])
print(df.min())
#df.to_csv('country_month_scale.csv',index=False)

# 确保数据按国家、年份和月份排序
data.sort_values(by=['country', 'year', 'month'], inplace=True)

# 创建一个字典来存储每个国家的矩阵
country_matrices = {}

# 遍历每个国家
for country, group in data.groupby('country'):
    # 创建一个完整的时间索引
    full_index = pd.date_range(start=f"2020-12",
                               end=f"2022-01",
                               freq='MS')

    # 将数据重建为完整的 DataFrame
    group_full = group.set_index(pd.to_datetime(group[['year', 'month']].assign(day=1)))
    #print(group_full.shape)
    group_full = group_full.reindex(full_index, method='ffill').fillna(method='bfill')  # 先向前再向后填充缺失值

    if group_full.isnull().sum().sum() > 0:
        print(f"{country} 有缺失值，考虑其他处理方法。")

    # 选择第四到第八列
    matrix = group_full.loc[:, ['affinity','wt','vac']].values
    country_matrices[country] = matrix

# 展平矩阵为向量
country_vectors = {country: matrix.flatten(order='F') for country, matrix in country_matrices.items()}

# 将字典转换为DataFrame，行是国家，列是展平后的向量元素
heatmap_df = pd.DataFrame.from_dict(country_vectors, orient='index')
# 保存
heat = heatmap_df.reset_index().rename(columns={'index':'country'})
heat.to_csv('results/vac_tongji/country_month_heat.csv',index=False)

ddic = {'country':[],'category':[]}
for k,v in categories.items():
    ddic['country'].append(k)
    ddic['category'].append(v)
dd = pd.DataFrame(ddic)
dd.to_csv('results/vac_tongji/annotation_country.csv',index=False)