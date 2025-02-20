import pandas as pd
from scipy import stats

df_wt = pd.read_csv('results/tongji_torch/combined_wt.csv')
df_wt['date'] = pd.to_datetime(df_wt['date'])
df_ina = pd.read_csv('results/tongji_torch/combined_ina.csv')
df_ina['date'] = pd.to_datetime(df_ina['date'])

# 获取所有列名为 "mean" 的列
mean_columns_wt = [col for col in df_wt.columns if 'mean' in col]
mean_columns_ina = [col for col in df_ina.columns if 'mean' in col]

df_wt = df_wt.loc[:, ['date'] + mean_columns_wt]
df_ina = df_ina.loc[:, ['date'] + mean_columns_ina]

# 选择日期为2023-01-01和2023-01-03的行
date1 = '2021-05-01'
date2 = '2021-07-01'

row_wt1 = df_wt[df_wt['date'] == date1].drop(columns=['date']).reset_index(drop=True)
row_wt2 = df_wt[df_wt['date'] == date2].drop(columns=['date']).reset_index(drop=True)
row_ina1 = df_ina[df_ina['date'] == date1].drop(columns=['date']).reset_index(drop=True)
row_ina2 = df_ina[df_ina['date'] == date2].drop(columns=['date']).reset_index(drop=True)

diff_wt1 = row_wt2 - row_wt1
diff_ina1 = row_ina2 - row_ina1

# 配对样本t检验
t_stat1, p_val1 = stats.ttest_rel(diff_wt1.T, diff_ina1.T)
print(t_stat1,p_val1)

date3 = '2021-12-01'
date4 = '2022-02-01'

row_wt3 = df_wt[df_wt['date'] == date3].drop(columns=['date']).reset_index(drop=True)
row_wt4 = df_wt[df_wt['date'] == date4].drop(columns=['date']).reset_index(drop=True)
row_ina3 = df_ina[df_ina['date'] == date3].drop(columns=['date']).reset_index(drop=True)
row_ina4 = df_ina[df_ina['date'] == date4].drop(columns=['date']).reset_index(drop=True)

diff_wt2 = row_wt4 - row_wt3
diff_ina2 = row_ina4 - row_ina3

t_stat2, p_val2 = stats.ttest_rel(diff_wt2.T, diff_ina2.T)
print(t_stat2,p_val2)

dd = pd.DataFrame({'wtp1':diff_wt1.values.flatten(),'inap1':diff_ina1.values.flatten(),'wtp2':diff_wt2.values.flatten(),'inap2':diff_ina2.values.flatten()})

dd.to_csv('results/diff_p1p2.csv',index=0)
