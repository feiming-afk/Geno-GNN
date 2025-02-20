import pandas as pd
import pylab as plt
import scipy.stats as stats

lis = ['aden_random_mrna_random','aden_random_inac_random','inac_random_mrna_random']
v_dic = {
    lis[1]:'Inactivated vs Adenovirus vaccine',
    lis[0]:'Adenovirus vs mRNA vaccine',
    lis[2]:'Inactivated vs mRNA vaccine'
}
res = {'pre': range(18), 'post': [], 'low_ci': [], 'upper_ci': [], 'nonsig': [], 'transition': [], 'fitness': [], 'vesus': []}
for l in lis:
    print(l)
    df = pd.read_csv(fr'D:\python_workspace\phenotype_v2\figures\TableS3\results\random\{l}\dumseq_p_values_all.csv')
    res['vesus'] += [v_dic[l]]*6
    print('Segment1!!')
    res['transition'] += ['t1']*3
    df1 = df[df['segment'] == 1]

    # 计算标准差
    res['fitness'].append('ACE2 affinity')
    mn = df1['affinity'].mean()

    # 计算标准误差
    sem = stats.sem(df1['affinity'])

    # 计算95%置信区间
    conf_interval = stats.t.interval(0.95, len(df1['affinity']) - 1, loc=mn, scale=sem)


    cnt1 = df1[df1['affinity'] > 0.1]['affinity'].count()
    res['post'].append(mn)
    res['nonsig'].append(cnt1)
    res['low_ci'].append(conf_interval[0])
    res['upper_ci'].append(conf_interval[1])

    # 计算标准差
    res['fitness'].append('WT convalescent')

    mn = df1['wt'].mean()

    sd = df1['wt'].std()
    # 计算标准误差
    sem = stats.sem(df1['wt'])

    # 计算95%置信区间
    conf_interval = stats.t.interval(0.95, len(df1['wt']) - 1, loc=mn, scale=sem)
    cnt2 = df1[df1['wt'] > 0.1]['wt'].count()
    res['post'].append(mn)
    res['nonsig'].append(cnt2)
    res['low_ci'].append(conf_interval[0])
    res['upper_ci'].append(conf_interval[1])
    # 计算标准差
    res['fitness'].append('WT inactive vaccine')
    mn = df1['ina'].mean()
    sd = df1['ina'].std()
    # 计算标准误差
    sem = stats.sem(df1['ina'])

    # 计算95%置信区间
    conf_interval = stats.t.interval(0.95, len(df1['ina']) - 1, loc=mn, scale=sem)
    cnt3 = df1[df1['ina'] > 0.1]['ina'].count()
    res['post'].append(mn)
    res['nonsig'].append(cnt3)
    res['low_ci'].append(conf_interval[0])
    res['upper_ci'].append(conf_interval[1])
    print('Segment2!!')
    res['transition'] += ['t2'] * 3
    df2 = df[df['segment'] == 2]
    # 计算标准差
    res['fitness'].append('ACE2 affinity')
    mn = df2['affinity'].mean()
    sd = df2['affinity'].std()
    # 计算标准误差
    sem = stats.sem(df2['affinity'])

    # 计算95%置信区间
    conf_interval = stats.t.interval(0.95, len(df2['affinity']) - 1, loc=mn, scale=sem)
    cnt1 = df2[df2['affinity'] > 0.1]['affinity'].count()
    res['post'].append(mn)
    res['nonsig'].append(cnt1)
    res['low_ci'].append(conf_interval[0])
    res['upper_ci'].append(conf_interval[1])
    # 计算标准差
    res['fitness'].append('WT convalescent')
    mn = df2['wt'].mean()
    sd = df2['wt'].std()
    # 计算标准误差
    sem = stats.sem(df2['wt'])

    # 计算95%置信区间
    conf_interval = stats.t.interval(0.95, len(df2['wt']) - 1, loc=mn, scale=sem)
    cnt2 = df2[df2['wt'] > 0.1]['wt'].count()
    res['post'].append(mn)
    res['nonsig'].append(cnt2)
    res['low_ci'].append(conf_interval[0])
    res['upper_ci'].append(conf_interval[1])
    # 计算标准差
    res['fitness'].append('WT inactive vaccine')
    mn = df2['ina'].mean()
    sd = df2['ina'].std()
    # 计算标准误差
    sem = stats.sem(df2['ina'])

    # 计算95%置信区间
    conf_interval = stats.t.interval(0.95, len(df2['ina']) - 1, loc=mn, scale=sem)
    cnt3 = df2[df2['ina'] > 0.1]['ina'].count()
    res['post'].append(mn)
    res['nonsig'].append(cnt3)
    res['low_ci'].append(conf_interval[0])
    res['upper_ci'].append(conf_interval[1])

for k,v in res.items():
    print(k)
    print(len(v))
ds = pd.DataFrame(res)
ds.to_csv('results/vac_tongji/random/p_values_diff.csv',index=False)

