import os
import os.path as osp
import numpy as np
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
    embeds = None
    node_types = []
    flagOut = True
    for seq in tqdm(seqs):
        embed = None
        node_type = []
        flagIn = True
        for char in seq:
            value = copy.deepcopy(vhse[char])
            node_type.append(vhse_id[char])
            if flagIn:
                embed = value
                flagIn = False
                continue
            embed = np.concatenate((embed, value), axis=0)
        node_types.append(node_type)
        if flagOut:
            embeds = embed.reshape(1, -1, 8)
            flagOut = False
            continue
        embeds = np.concatenate((embeds, embed.reshape(1, -1, 8)), axis=0)
    
    node_types = np.array(node_types)

    print(embeds.shape)
    print(node_types.shape)

    return embeds, node_types


def get_data(raw_dir, name, mut, linear_label):
    embeds_path = osp.join(raw_dir, f'{name}_embeds.npy')
    bds_path = osp.join(raw_dir, f'{name}_bind.npy')
    node_types_path = osp.join(raw_dir, f'{name}_node_types.npy')
    if os.path.exists(bds_path):
        embeds = np.load(embeds_path)
        bds = np.load(bds_path)
        node_types = np.load(node_types_path)
        # B, N, D = seqs.shape
    else:
        seqs_bds_path = osp.join(raw_dir, f'{name}_seqs_bds_{mut}.csv')
        seqs = []
        bds = []
        with open(seqs_bds_path, mode='r') as f:
            for line in f:
                lis = line.split(',')
                if linear_label:
                    seq, bind = lis[0], lis[2]
                else:
                    seq, bind = lis[0], lis[1]
                bind = bind[:-1]
                seqs.append(seq)
                bds.append(bind)
        bds = np.array(bds, dtype=float)
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

class GENODataset(InMemoryDataset):
    def __init__(self, root, name='all', mut=1, linear_label=False, transform=None, pre_transform=None, pre_filter=None):
        self.name = name
        self.mut = mut
        self.linear_label=linear_label
        assert name in ['all', 'noBA2', 'EscapeV2_BA1', 'EscapeV2_BA2', 'Alpha', 'Beta', 'Delta', 'Eta', 'WHv1', 'WHv2'], 'Invalid process!'
        if linear_label:
            assert mut == 2
            self.dataset_name = f'GENO_M{mut}_{name}'
        else:
            self.dataset_name = f'GENO_M{mut}_{name}'

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_dir(self) -> str:
        # 定义原始数据目录
        return osp.join(self.root, self.dataset_name, 'raw')

    @property
    def processed_dir(self) -> str:
        # 定义处理过的数据目录
        return osp.join(self.root, self.dataset_name, 'processed')

    @property
    def raw_file_names(self):
        # 定义原始文件名列表
        return [f'{self.name}_Embeds.npy', f'{self.name}_bds.npy']

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
        data_list = get_data(self.raw_dir, self.name, self.mut, self.linear_label)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

if __name__ == '__main__':
    # names = ['EscapeV2_BA1', 'EscapeV2_BA2', 'Alpha', 'Beta', 'Delta', 'Eta', 'WHv1', 'WHv2', 'all', 'noBA2']
    # for name in names:
    #     dataset = GENODataset(root='./data/', name=name, mut=1)
    #     print(f"Sucessfully load GENO {name} mut1.")
    #     print('\t', dataset)
    #     print('\t', dataset[0])
    names = ['EscapeV2_BA1', 'EscapeV2_BA2', 'Alpha', 'Beta', 'Delta', 'Eta', 'WHv1', 'all']
    for name in names:
        dataset = GENODataset(root='./data/', name=name, mut=2, linear_label=True)
        print(f"Sucessfully load GENO {name} mut2.")
        print('\t', dataset)
        print('\t', dataset[0])
