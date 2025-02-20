import pandas as pd

df = pd.read_csv('test_results/site_test/site_test.csv')
affinity = pd.read_csv('../data/final_variant_scores.csv')
wt = pd.read_csv('../data/escape_v4/WT convalescents.csv')
vac = pd.read_csv('../data/escape_v4/WT vaccinees.csv')
ba1 = pd.read_csv('../data/escape_v4/BA.1 convalescents.csv')
ba2 = pd.read_csv('../data/escape_v4/BA.2 convalescents.csv')
ba5 = pd.read_csv('../data/escape_v4/BA.5 convalescents.csv')

df['site'] = df['site'] + 331
merged_df = pd.merge(df, wt, on=['site', 'mutation'], how='left')
merged_df = pd.merge(merged_df, vac, on=['site', 'mutation'], how='left', suffixes=('_wt', ''))
merged_df = pd.merge(merged_df, ba1, on=['site', 'mutation'], how='left', suffixes=('_vac', ''))
merged_df = pd.merge(merged_df, ba2, on=['site', 'mutation'], how='left', suffixes=('_ba1', ''))
merged_df = pd.merge(merged_df, ba5, on=['site', 'mutation'], how='left', suffixes=('_ba2', '_ba5'))

affinity = affinity[affinity['target'] == 'Wuhan-Hu-1'][['position', 'mutant', 'bind']].rename(columns={'position':'site','mutant':'mutation'})
merged_df = pd.merge(merged_df, affinity, on=['site', 'mutation'], how='left')

merged_df.to_csv('test_results/site_test/site_test_expr.csv', index=False)

site_test = merged_df.groupby('site').mean().reset_index().rename(columns={'affinity':'affinity_pred','bind':'affinity_expr',
                                                                               'wt':'wt_pred','mut_escape_wt':'wt_expr',
                                                                           'vac':'vac_pred','mut_escape_vac':'vac_expr',
                                                                           'ba1':'ba1_pred','mut_escape_ba1':'ba1_expr',
                                                                           'ba2':'ba2_pred','mut_escape_ba2':'ba2_expr',
                                                                           'ba5':'ba5_pred','mut_escape_ba5':'ba5_expr',})

site_test.to_csv('test_results/site_test/site_test_expr_mean.csv', index=False)
