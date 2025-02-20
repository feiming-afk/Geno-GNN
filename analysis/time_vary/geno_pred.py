import numpy as np
import torch
from torch_geometric.data import Data
from concurrent.futures import ProcessPoolExecutor
import os
import sys
sys.path.append('../Geno_GNN_train_test/function')
from models import REGNN

def VHSE_featurize(seqs):
    from tqdm import tqdm
    vhse = {}
    vhse_id = {}
    vhselis = ['A,0.15,-1.11,-1.35,-0.92,0.02,-0.91,0.36,-0.48', 'R,-1.47,1.45,1.24,1.27,1.55,1.47,1.3,0.83',
               'N,-0.99,0,-0.37,0.69,-0.55,0.85,0.73,-0.8', 'D,-1.15,0.67,-0.41,-0.01,-2.68,1.31,0.03,0.56',
               'C,0.18,-1.67,-0.46,-0.21,0,1.2,-1.61,-0.19', 'Q,-0.96,0.12,0.18,0.16,0.09,0.42,-0.2,-0.41',
               'E,-1.18,0.4,0.1,0.36,-2.16,-0.17,0.91,0.02', 'G,-0.2,-1.53,-2.63,2.28,-0.53,-1.18,2.01,-1.34',
               'H,-0.43,-0.25,0.37,0.19,0.51,1.28,0.93,0.65', 'I,1.27,-0.14,0.3,-1.8,0.3,-1.61,-0.16,-0.13',
               'L,1.36,0.07,0.26,-0.8,0.22,-1.37,0.08,-0.62', 'K,-1.17,0.7,0.7,0.8,1.64,0.67,1.63,0.13',
               'M,1.01,-0.53,0.43,0,0.23,0.1,-0.86,-0.68', 'F,1.52,0.61,0.96,-0.16,0.25,0.28,-1.33,-0.2',
               'P,0.22,-0.17,-0.5,0.05,-0.01,-1.34,-0.19,3.56', 'S,-0.67,-0.86,-1.07,-0.41,-0.32,0.27,-0.64,0.11',
               'T,-0.34,-0.51,-0.55,-1.06,-0.06,-0.01,-0.79,0.39', 'W,1.5,2.06,1.79,0.75,0.75,-0.13,-1.01,-0.85',
               'Y,0.61,1.6,1.17,0.73,0.53,0.25,-0.96,-0.52', 'V,0.76,-0.92,-0.17,-1.91,0.22,-1.4,-0.24,-0.03']
    cnt = 0
    for i in vhselis:
        lis = i.split(',')
        vhse[lis[0][-1]] = np.array([float(k) for k in lis[1:]]).reshape(1, -1)
        vhse_id[lis[0][-1]] = cnt
        cnt += 1

    embeds = []
    node_types = []
    for seq in tqdm(seqs):
        if '*' in seq:
            print(seq)
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

def featurelize_parallel(seqs,n):
    with ProcessPoolExecutor(max_workers=n) as executor:
        results = executor.map(VHSE_featurize, np.array_split(seqs, n))
        embeds, node_types = zip(*results)
    return np.concatenate(embeds), np.concatenate(node_types)

def get_data(seqs,n):
    embeds, node_types = featurelize_parallel(seqs, n)
    print(embeds.shape), print(node_types.shape)
    B, N, D = embeds.shape

    # get edge_index
    row = list(range(0, N - 1))
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
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
        data_list.append(data)
    return data_list

def pred(model, dataloader, device):
    model.eval()
    outputs = []
    for data in dataloader:
        data = data.to(device)
        output = model(data, return_emb = False)
        output = output.reshape(-1)
        outputs.append(output.cpu().detach().numpy())
    outputs = np.concatenate(outputs)
    print(outputs.shape)
    return outputs

def predcess(df, dataloder):
    mdl_dic = {}
    device = f'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(device)

    model_bds = REGNN(8, 256, 1, 2, 0., graph_pooling='concat',
                        norm='none', scaling_factor=10., no_re=False).to(device)
    model_bds.load_state_dict(torch.load(f'../Geno_GNN_train_test/models/affinity.pt'))
    mdl_dic['affinity'] = model_bds

    model_wt = REGNN(8, 256, 1, 2, 0., graph_pooling='concat',
                       norm='none', scaling_factor=1., no_re=False).to(device)
    model_wt.load_state_dict(torch.load(f'../Geno_GNN_train_test/models/escape_WT.pt'))
    mdl_dic['wt'] = model_wt

    model_ina = REGNN(8, 256, 1, 2, 0., graph_pooling='concat',
                        norm='none', scaling_factor=1., no_re=False).to(device)
    model_ina.load_state_dict(torch.load(f'../Geno_GNN_train_test/models/escape_VAC.pt'))
    mdl_dic['vac'] = model_ina

    model_ba1 = REGNN(8, 256, 1, 2, 0., graph_pooling='concat',
                        norm='none', scaling_factor=1., no_re=False).to(device)
    model_ba1.load_state_dict(torch.load(f'../Geno_GNN_train_test/models/escape_BA1.pt'))
    mdl_dic['ba1'] = model_ba1

    model_ba2 = REGNN(8, 256, 1, 2, 0., graph_pooling='concat',
                        norm='none', scaling_factor=1., no_re=False).to(device)
    model_ba2.load_state_dict(torch.load(f'../Geno_GNN_train_test/models/escape_BA2.pt'))
    mdl_dic['ba2'] = model_ba2

    model_ba5 = REGNN(8, 256, 1, 2, 0., graph_pooling='concat',
                        norm='none', scaling_factor=1., no_re=False).to(device)
    model_ba5.load_state_dict(torch.load(f'../Geno_GNN_train_test/models/escape_BA5.pt'))
    mdl_dic['ba5'] = model_ba5

    for k, v in mdl_dic.items():
        outputs = pred(v, dataloder, device)
        df[k] = outputs

    return df