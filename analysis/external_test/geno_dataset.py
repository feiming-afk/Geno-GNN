import os
import os.path as osp
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset, download_url
import pandas as pd
import re
from Bio import SeqIO
import copy

def VHSE_featurize(seqs):
    from tqdm import tqdm
    vhse = {}
    vhse_id = {}
    with open(osp.join('funcdata', 'VHSE.csv'), mode='r', encoding='utf-8') as f:
        vhselis = f.read().split('\n')[:-1]
        cnt = 0 
        for i in vhselis:
            lis = i.split(',')
            vhse[lis[0][-1]] = np.array([float(k) for k in lis[1:]]).reshape(1, -1)
            vhse_id[lis[0][-1]] = cnt
            cnt += 1
    embeds = []
    node_types = []
    for seq in tqdm(seqs):
        embed = []
        node_type = []
        for char in seq:
            value = vhse[char].tolist()
            node_type.append(vhse_id[char])
            embed.append(value)

        node_types.append(node_type)
        embeds.append(embed)
    
    node_types = np.array(node_types)
    embeds = np.array(embeds, dtype=np.float32).reshape(len(embeds), -1, 8)

    print(embeds.shape)
    print(node_types.shape)

    return embeds, node_types

def get_ba1_seqs(raw_dir):
    raw_data_path = osp.join(raw_dir,'cleaned_Kds_RBD_ACE2.tsv')
    ml_path = osp.join(raw_dir, 'strange', 'mutation_list.csv')
    B_path = osp.join(raw_dir, 'strange', 'B')
    seqs = []
    binds = []
    df = pd.read_csv(raw_data_path, sep='\t', low_memory=False)[['geno','log10Kd_pinned']]
    ml = pd.read_csv(ml_path)['mut_name']
    for i in df.index:
        sc = df.loc[i,'log10Kd_pinned']
        if pd.isnull(sc):
            continue
        bb = str(df.loc[i,'geno']).zfill(15)
        starts = [each.start() for each in re.finditer("1",bb)]
        muts = ml[starts]
        with open(B_path, 'r') as f:
            f.readline()
            wt = [j for j in ''.join(f.read().split('\n')).strip()]
        for k in muts:
            org = k[0]
            mut = k[-1]
            pos = int(k[1:-1])-1
            wt[pos] = mut
        seqs.append(''.join(wt[330:531]))
        binds.append(sc)
    return seqs,binds

def get_omicron_seqs(raw_dir):
    raw_data_path = osp.join(raw_dir, 'affinity_omicron.csv')
    df = pd.read_csv(raw_data_path)[['target', 'wildtype', 'position', 'mutant', 'mutation', 'bind']]

    seq_path = osp.join(raw_dir, 'omicron_seqs.csv')
    df_seq = pd.read_csv(seq_path)

    seq_dic = {}
    seqs = []
    binds = []

    # 创建序列字典
    for index, row in df_seq.iterrows():
        target = row['target']
        if target not in seq_dic:
            seq_dic[target] = []
        wt = row['wildtype']
        seq_dic[target].append(wt)

    # 获取突变序列和结合亲和力
    for index, row in df.iterrows():
        target = row['target']
        wild_seq = copy.deepcopy(seq_dic[target])
        mut = row['mutant']
        bind = row['bind']
        pos = int(row['position']) - 331  # 根据位置调整索引
        if pd.isnull(bind):
            print('空值跳过!!',end='')
            continue
        # 替换野生型序列中的突变
        wild_seq[pos] = mut
        seqs.append(''.join(wild_seq))
        binds.append(bind)

    return seqs, binds

def get_escape_seqs(raw_dir, name):
    root_dir = osp.dirname(osp.dirname(raw_dir))
    nt50_dir = osp.join(root_dir, 'nt50')
    df = pd.read_csv(osp.join(root_dir, 'nt50.csv'), index_col=0)
    seqs = []
    bds = []
    folder_list = os.listdir(nt50_dir)
    for folder in folder_list:
        for f in os.listdir(os.path.join(nt50_dir, folder)):
            if not str(f).endswith('fasta'):
                continue
            relative_path = os.path.join(nt50_dir, folder, str(f))
            fa_seq = [str(SeqIO.read(relative_path, "fasta").seq)[330:531]]
            seqs.append(''.join(fa_seq))
            bd = df.loc[folder, name]
            bds.append(bd)
    return seqs,bds

def get_data_update(raw_dir, name):
    embeds_path = osp.join(raw_dir, f'{name}_embeds.npy')
    bds_path = osp.join(raw_dir, f'{name}_y.npy')
    node_types_path = osp.join(raw_dir, f'{name}_node_types.npy')
    # if os.path.exists(bds_path):
    if os.path.exists(bds_path):
        embeds = np.load(embeds_path)
        bds = np.load(bds_path)
        node_types = np.load(node_types_path)
        # B, N, D = seqs.shape
    elif name == 'affinity':
        seqs,bds = get_ba1_seqs(raw_dir)
        bds = np.array(bds, dtype=float)
        embeds, node_types = VHSE_featurize(seqs)

        np.save(embeds_path, embeds)
        np.save(bds_path, bds)
        np.save(node_types_path, node_types)
    elif name == 'affinity_Omicron':
        seqs, bds = get_omicron_seqs(raw_dir)
        bds = np.array(bds, dtype=float)
        embeds, node_types = VHSE_featurize(seqs)

        np.save(embeds_path, embeds)
        np.save(bds_path, bds)
        np.save(node_types_path, node_types)
    else:
        seqs,bds = get_escape_seqs(raw_dir,name)
        bds = np.array(bds, dtype=float)
        embeds, node_types = VHSE_featurize(seqs)

        np.save(embeds_path, embeds)
        np.save(bds_path, bds)
        np.save(node_types_path, node_types)
    print(embeds.shape, node_types.shape)
    print(np.any(np.isnan(bds)))
    B, N, D = embeds.shape

    # get edge_index
    row = list(range(0, N-1))
    col = list(range(1, N))
    edge_index_row = row + col
    edge_index_col = col + row
    edge_index = [edge_index_row, edge_index_col]

    data_list = []
    for i in range(B):
        x = torch.FloatTensor(embeds[i])
        pos = torch.LongTensor(node_types[i])
        edge_index = torch.LongTensor(edge_index)
        node_type_row = pos[edge_index_row]
        node_type_col = pos[edge_index_col]
        edge_attr = node_type_row * 20 + node_type_col
        edge_attr = torch.LongTensor(edge_attr)
        # print(edge_attr.min(), edge_attr.max())
        y = torch.FloatTensor([bds[i]])
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, pos=pos)
        data_list.append(data)
    return data_list

class GENODataset_update(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None,
                 pre_filter=None):
        self.name = name

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        # 定义原始数据目录
        return osp.join(self.root, self.name, 'data')

    @property
    def processed_dir(self) -> str:
        # 定义处理过的数据目录
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self):
        # 定义原始文件名列表
        return [f'{self.name}_embeds.npy', f'{self.name}_y.npy']

    @property
    def processed_file_names(self):
        # 定义处理过的文件名列表
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        # download_url(url, self.raw_dir)
        pass

    def process(self):
        # 处理方法
        # Read data into huge `Data` list.
        data_list = get_data_update(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

if __name__ == '__main__':
    dataset = GENODataset_update(root='testdata/affinity_variant',name='affinity')
    dataset = GENODataset_update(root='testdata/affinity_variant',name='affinity_Omicron')

    names = ['CoronaVac', 'BA1', 'BA2', 'BA5']
    for nn in names:
        dataset = GENODataset_update(root='testdata/escape/data', name=nn)
