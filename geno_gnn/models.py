import torch
import time
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GraphConv, GCNConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, Set2Set
# from pooling.globalattpool import GlobalAttention
from regnn_layers import REGCNConv
import random
import numpy as np


class GNN(torch.nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel, num_gnn_layers, dropout=0., graph_pooling='sum'):
        super(GNN, self).__init__()
        self.dropout = dropout
        self.num_gnn_layers = num_gnn_layers
        self.graph_pooling = graph_pooling
        self.convs = torch.nn.ModuleList()
        for i in range(num_gnn_layers):
            in_dim = in_channel if i == 0 else hidden_channel
            out_dim = hidden_channel
            self.convs.append(GraphConv(in_dim, out_dim))

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling in ['readout', 'concat']:
            pass
        elif graph_pooling == "set2set":
            self.pool = Set2Set(hidden_channel, processing_steps=2)
        # elif args.graph_pooling == "subgraphpool":
        #     self.pool = SubGraphPool(self.hidden, self.gp_heads)
        else:
            raise ValueError("Invalid graph pooling type.")

        if graph_pooling == 'readout':
            self.lin1 = torch.nn.Linear(hidden_channel * 2, 64)
        elif graph_pooling == 'concat':
            self.lin1 = torch.nn.Linear(hidden_channel * 201, 64)
        else:
            self.lin1 = torch.nn.Linear(hidden_channel, 64)
        self.lin2 = torch.nn.Linear(64, 32)
        self.lin3 = torch.nn.Linear(32, out_channel)

        self.reset_parameters()
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x_add = None
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if x_add is None:
                if self.graph_pooling == 'readout':
                    x_add = torch.cat([gap(x, batch), gmp(x, batch)], dim=1)
                elif self.graph_pooling in ['sum', 'mean', 'max']:
                    x_add = self.pool(x, batch)
            else:
                if self.graph_pooling == 'readout':
                    x_add += torch.cat([gap(x, batch), gmp(x, batch)], dim=1)
                elif self.graph_pooling in ['sum', 'mean', 'max']:
                    x_add += self.pool(x, batch)
        if self.graph_pooling == 'concat':
            x_add = torch.reshape(x, (-1, x.shape))
        x = F.dropout(x_add, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin2(x))
        x = self.lin3(x)

        return x
    

class REGNN(torch.nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel, num_gnn_layers, dropout=0., graph_pooling='sum', 
            norm='none', scaling_factor=1.0, no_re=False):
        super(REGNN, self).__init__()
        self.dropout = dropout
        self.num_gnn_layers = num_gnn_layers
        self.graph_pooling = graph_pooling
        self.hidden_channel = hidden_channel
        self.convs = torch.nn.ModuleList()
        for i in range(num_gnn_layers):
            in_dim = in_channel if i == 0 else hidden_channel
            out_dim = hidden_channel
            self.convs.append(REGCNConv(in_dim, out_dim, 20, 20*20, scaling_factor=scaling_factor, use_norm=norm, no_re=no_re))

        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling in ['readout', 'concat']:
            pass
        elif graph_pooling == "set2set":
            self.pool = Set2Set(hidden_channel, processing_steps=2)
        # elif args.graph_pooling == "subgraphpool":
        #     self.pool = SubGraphPool(self.hidden, self.gp_heads)
        else:
            raise ValueError(f"Invalid graph pooling type: {graph_pooling}.")

        if graph_pooling == 'readout':
            self.lin1 = torch.nn.Linear(hidden_channel * 2, 64)
        elif graph_pooling == 'concat':
            self.lin1 = torch.nn.Linear(hidden_channel * 201, 64)
        else:
            self.lin1 = torch.nn.Linear(hidden_channel, 64)
        self.lin2 = torch.nn.Linear(64, 32)
        self.lin3 = torch.nn.Linear(32, out_channel)

        self.reset_parameters()
    
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, data, return_emb=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        edge_attr, pos = data.edge_attr, data.pos
        x_add = None
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr, pos)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if x_add is None:
                if self.graph_pooling == 'readout':
                    x_add = torch.cat([gap(x, batch), gmp(x, batch)], dim=1)
                elif self.graph_pooling in ['sum', 'mean', 'max']:
                    x_add = self.pool(x, batch)
            else:
                if self.graph_pooling == 'readout':
                    x_add += torch.cat([gap(x, batch), gmp(x, batch)], dim=1)
                elif self.graph_pooling in ['sum', 'mean', 'max']:
                    x_add += self.pool(x, batch)
        if self.graph_pooling == 'concat':
            x_add = x.reshape((-1, 201 * self.hidden_channel))

        x = F.dropout(x_add, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        emb = self.lin2(x)
        out = F.relu(emb)
        out = self.lin3(out)

        if return_emb:
            return out, emb

        return out
        
    

class old_GNN(torch.nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel, num_gnn_layers, dropout=0., graph_pooling='sum'):
        super(old_GNN, self).__init__()
        self.dropout = dropout
        self.num_gnn_layers = num_gnn_layers
        self.convs = torch.nn.ModuleList()
        for i in range(num_gnn_layers):
            in_dim = in_channel if i == 0 else hidden_channel
            out_dim = hidden_channel
            self.convs.append(GraphConv(in_dim, out_dim))
        
        # self.pool1 = TopKPooling(64, ratio=0.8)
        # self.pool2 = TopKPooling(64, ratio=0.8)
        # self.pool3 = TopKPooling(64, ratio=0.8)

        self.lin1 = torch.nn.Linear(hidden_channel*2, 64)
        self.lin2 = torch.nn.Linear(64, 32)
        self.lin3 = torch.nn.Linear(32, out_channel)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.convs[0](x, edge_index))
        # x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gap(x, batch), gmp(x, batch)], dim=1)

        x = F.relu(self.convs[1](x, edge_index))
        # x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gap(x, batch), gmp(x, batch)], dim=1)

        x = F.relu(self.convs[2](x, edge_index))
        # x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        x3 = torch.cat([gap(x, batch), gmp(x, batch)], dim=1)

        x = x1 + x2 + x3

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = self.lin3(x)

        return x 