import torch
import torch.nn as nn
# from torch_sparse import matmul
import torch_geometric.nn as gnn
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import reset
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch_geometric.utils.loop import add_self_loops, remove_self_loops
# from torch_sparse import SparseTensor

from .BaseGNN import GNNBasic
from .MolEncoders import AtomEncoder, BondEncoder


class GCNEConv(gnn.MessagePassing):
    def __init__(self, in_channels, out_channels, config, bias=True, mol=False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GCNEConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self._cached_edge_index = None
        self._cached_adj_t = None

        # self.lin = Linear(in_channels, out_channels, bias=False, weight_initializer='glorot')

        self.bond_encoder = BondEncoder(in_channels, config) if mol else None

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels))
        self.eps = nn.Parameter(torch.Tensor([0]))



        # if bias:
        #     self.bias = nn.Parameter(torch.Tensor(out_channels))
        # else:
        #     self.register_parameter('bias', None)

        # self.reset_parameters()

    def reset_parameters(self):
        self.lin.reset_parameters()
        gnn.inits.zeros(self.bias)
        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x, edge_index, edge_attr):

        if self.bond_encoder is not None:
            edge_attr = self.bond_encoder(edge_attr)

        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_attr))

        # x = self.lin(x)
        #
        # out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=None)
        # if self.bias is not None:
        #     out += self.bias

        return out

    def message(self, x_j, edge_attr):
        return nn.functional.relu(x_j + edge_attr)
        # return x_j if edge_attr is None else nn.functional.relu(x_j + edge_attr)

    # def message_and_aggregate(self, adj_t, x):
    #     return torch.matmul(adj_t, x, reduce=self.aggr)


class GCN(GNNBasic):
    def __init__(self, config, **kwargs):
        super(GCN, self).__init__(config)
        self.config = config
        self.num_layer = config['model']['num_layer']
        self.model_level = config['model']['model_level']
        self.pool_type = config['model']['global_pool']

        if config['dataset']['dataset_type'] == 'mol':
            self.edge_feat = True
            self.encoder = AtomEncoder(config['model']['dim_hidden'], config)
            self.convs = nn.ModuleList(
                [
                    GCNEConv(config['model']['dim_hidden'], config['model']['dim_hidden'], config, mol=True)
                    for _ in range(self.num_layer)
                ]
            )
        else:
            self.edge_feat = False
            self.convs = nn.ModuleList()
            self.convs.append(
                GCNEConv(config['dataset']['dim_node'], config['model']['dim_hidden'], config))

            self.convs = self.convs.extend(
                [
                    GCNEConv(config['model']['dim_hidden'], config['model']['dim_hidden'], config)
                    for _ in range(self.num_layer - 1)
                ]
            )

        self.batch_norms = nn.ModuleList(
            [
                nn.BatchNorm1d(config['model']['dim_hidden'])
                for _ in range(self.num_layer)
            ]
        )

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config['model']['dropout_rate'])

        if config['model']['model_level'] == 'node':
            self.pool = None
        elif self.pool_type == 'mean':
            self.pool = gnn.global_mean_pool
        elif self.pool_type == 'max':
            self.pool = gnn.global_max_pool
        else:
            self.pool = None
            raise Warning('Missing specific pool type!')

        self.classifier = nn.Linear(config['model']['dim_hidden'], config['dataset']['num_classes'])

    def get_node_repr(self, x, edge_index, edge_attr):

        x = self.encoder(x) if self.encoder is not None else x

        for i in range(self.num_layer):
            x = self.convs[i](x, edge_index, edge_attr) if self.edge_feat else self.convs[i](x, edge_index)
            x = self.batch_norms[i](x)
            if i < self.num_layer - 1:
                x = self.relu(x)
            x = self.dropout(x)
        return x

    def forward(self, *args, **kwargs):
        if self.edge_feat:
            x, edge_index, edge_attr, batch, batch_size = self.arguments_read(*args, edge_feat=self.edge_feat, **kwargs)
        else:
            x, edge_index, batch, batch_size = self.arguments_read(*args, **kwargs)
            edge_attr = None
        kwargs.pop('batch_size', 'not found')
        node_feat = self.get_node_repr(x, edge_index, edge_attr)
        if self.pool is None:
            return self.classifier(x)
        else:
            return self.classifier(self.pool(node_feat, batch, batch_size))
