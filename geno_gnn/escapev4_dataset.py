import os
import os.path as osp
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_sparse import coalesce
from sklearn.model_selection import train_test_split
from torch_geometric.data import InMemoryDataset, download_url
import copy

def VHSE_featurize(seqs, raw_dir):
    from tqdm import tqdm
    vhse = {}
    vhse_id = {}
    with open(osp.join(raw_dir, 'VHSE.csv'), mode='r', encoding='utf-8') as f:
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


def get_data(raw_dir, name):
    embeds_path = osp.join(raw_dir, f'{name}_embeds.npy')
    bds_path = osp.join(raw_dir, f'{name}_bind.npy')
    node_types_path = osp.join(raw_dir, f'{name}_node_types.npy')
    if os.path.exists(bds_path):
        embeds = np.load(embeds_path)
        bds = np.load(bds_path)
        node_types = np.load(node_types_path)
    else:
        seqs_bds_path = osp.join(raw_dir, f'{name}.csv')
        wt_path = osp.join(raw_dir, 'B')

        df = pd.read_csv(seqs_bds_path, low_memory=False)
        with open(wt_path, 'r') as f:
            wt_lines = f.readlines()
            wt_seq = ''.join(line.strip() for line in wt_lines[1:])
            wt = [char for char in wt_seq][330:531]
        seqs = []
        bds = df['mut_escape'].to_numpy(dtype=float)
        for _,row in df.iterrows():
            site = int(row['site']) - 331
            new_wt = wt[:site] + list(row['mutation']) + wt[site + 1:]
            seqs.append(new_wt)

        embeds, node_types = VHSE_featurize(seqs, raw_dir)

        np.save(embeds_path, embeds)
        np.save(bds_path, bds)
        np.save(node_types_path, node_types)

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

class EscapeV4Dataset(InMemoryDataset):
    def __init__(self, root, name, transform=None, pre_transform=None, pre_filter=None):
        self.name = name

        self.dataset_name = f'EscapeV4_{name}'

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.dataset_name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.dataset_name, 'processed')

    @property
    def raw_file_names(self):
        return [f'{self.name}_Embeds.npy', f'{self.name}_bds.npy']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        # download_url(url, self.raw_dir)
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = get_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

if __name__ == '__main__':
    names = ['WT', 'VAC', 'BA1', 'BA2', 'BA5']
    for name in names:
        dataset = EscapeV4Dataset(root='../traindata/', name=name)
        print(f"Sucessfully load Dataset_EscapeV4_{name}.")
        print('\t', dataset)
        print('\t', dataset[0])
        print('\t', dataset.num_classes)
        print('\t', dataset.num_features)
