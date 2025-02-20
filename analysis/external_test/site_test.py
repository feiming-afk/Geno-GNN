from torch_geometric.loader import DataLoader
import pandas as pd
import copy
from geno_pred import *

amino = ['A','R','N','D','C','Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']

def mut_yield(seq_lis):
    seqs = []
    for i in range(201):
        for j in amino:
            ss = copy.deepcopy(seq_lis)
            ss[i] = j
            seqs.append(''.join(ss))
    return seqs

def site_fitness():
    with open('data/B','r') as f:
        f.readline()
        wt = [j for j in ''.join(f.read().split('\n')).strip()][330:531]
    seqs = mut_yield(wt)
    n = 1
    dataset = get_data(seqs, n)
    loader = DataLoader(dataset, batch_size=5000, shuffle=False)

    res = {'wildtype':['X']*4020, 'site':[0]*4020, 'mutation':['X']*4020}
    df = pd.DataFrame(res)
    df['wildtype'] = 'X'
    df['site'] = 0
    df['mutation'] = 'X'
    k = 0;v = 0
    for i in df.index:
        if k == 20:
            k = 0
            v += 1
        df.loc[i,'wildtype'] = wt[v]
        df.loc[i,'site'] =  v
        df.loc[i,'mutation'] = amino[k]
        k += 1

    df = predcess(df, loader)
    return df

if __name__ == '__main__':
    df = site_fitness()
    df.to_csv('test_results/site_test/site_test.csv', index = False)