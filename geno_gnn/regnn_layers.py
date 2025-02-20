import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import ModuleList, Linear, ModuleDict, Parameter, init, ParameterDict
from torch_geometric.nn import MessagePassing, GATv2Conv
from utils import weighted_degree, get_self_loop_index, softmax


class REGCNConv(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_node_types,
                 num_edge_types,
                 scaling_factor=100.,
                 dropout=0., 
                 use_softmax=False,
                 residual=True,
                 use_norm=None,
                 no_re=False):
        super(REGCNConv, self).__init__(aggr='mean')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_node_types = num_node_types
        self.num_edge_types = num_edge_types
        self.use_softmax = use_softmax
        self.dropout = dropout
        self.residual = residual
        self.use_norm = use_norm

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        if self.residual:
            # self.weight_root = self.weight 
            self.weight_root = Parameter(torch.Tensor(in_channels, out_channels))
        self.bias = Parameter(torch.Tensor(out_channels))
        rw_dim = self.num_edge_types + num_node_types
        if no_re:
            self.relation_weight = Parameter(torch.Tensor(rw_dim), requires_grad=False)
        else:
            self.relation_weight = Parameter(torch.Tensor(rw_dim), requires_grad=True)
        self.scaling_factor = scaling_factor

        if self.use_norm  == 'bn':
            self.norm = torch.nn.BatchNorm1d(out_channels)
        elif self.use_norm == 'ln':
            self.norm = torch.nn.LayerNorm(out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_uniform_(self.weight)
        if self.residual:
            init.xavier_uniform_(self.weight_root)
        init.zeros_(self.bias)
        init.constant_(self.relation_weight, 1.0 / self.scaling_factor)
        if self.use_norm in ['bn', 'ln']:
            self.norm.reset_parameters()

    def forward(self, x, edge_index, edge_type, target_node_type, return_weights=False):
        # shape of x: [N, in_channels]
        # shape of edge_index: [2, E]
        # shape of edge_type: [E]
        # shape of e_feat: [E, edge_tpes+node_types]

        
        # add self-loops to edge_index and edge_type
        num_nodes = target_node_type.size(0)
        loop_index = torch.arange(0, num_nodes, dtype=torch.long, device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, loop_index], dim=1)
        edge_type = torch.cat([edge_type, target_node_type+self.num_edge_types], dim=0)

        edge_type = edge_type.view(-1, 1)
        e_feat = torch.zeros(edge_type.shape[0], self.num_edge_types+self.num_node_types, device=edge_type.device).scatter_(1, edge_type, 1.0)

        x_src, x_target = x, x
        x_src = torch.matmul(x_src, self.weight)
        if self.residual:
            x_target = torch.matmul(x_target, self.weight_root)
        else:
            x_target = torch.matmul(x_target, self.weight)
        x = (x_src, x_target)

        # Cal edge weight according to its relation type
        relation_weight = self.relation_weight * self.scaling_factor
        relation_weight = F.leaky_relu(relation_weight)
        edge_weight = torch.matmul(e_feat, relation_weight.reshape(-1,1)).reshape(-1)  # [E]

        # Compute GCN normalization
        row, col = edge_index
        # self.use_softmax = True
        if self.use_softmax:
            ew = softmax(edge_weight, col)
        else:
            # mean aggregator
            deg = weighted_degree(col, edge_weight, x_target.size(0), dtype=x_target.dtype) #.abs()
            deg_inv = deg.pow(-1.0)
            norm = deg_inv[col]
            ew = edge_weight * norm
        
        # ew = F.dropout(ew, p=self.dropout, training=self.training)
        out = self.propagate(edge_index, x=x, ew=edge_weight)

        if self.residual:
            out += x_target
        
        if self.use_norm in ['bn', 'ln']:
            out = self.norm(out)

        if return_weights:
            return out, ew, relation_weight
        else:
            return out

    def message(self, x_j, ew):

        return ew.view(-1, 1) * x_j

    def update(self, aggr_out):

        aggr_out = aggr_out + self.bias

        return aggr_out
    