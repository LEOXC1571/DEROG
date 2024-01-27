import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from torch_geometric.nn.inits import reset
from torch_geometric.typing import OptPairTensor, Adj, OptTensor, Size
from torch_geometric.utils.loop import add_self_loops, remove_self_loops
from torch_sparse import SparseTensor

from .BaseGNN import GNNBasic, BasicEncoder
from .Classifiers import Classifier
from .MolEncoders import AtomEncoder, BondEncoder
from torch.nn import Identity


class MLP(nn.Sequential):
    def __init__(self, channels, dropout, config, bias=True, bn=False):
        m = []
        for i in range(1, len(channels)):
            m.append(nn.Linear(channels[i - 1], channels[i], bias))

            if i < len(channels) - 1:
                if bn:
                    m.append(nn.BatchNorm1d(channels[i]))
                else:
                    m.append(gnn.InstanceNorm(channels[i]))

                m.append(nn.ReLU())
                m.append(nn.Dropout(dropout))

        super(MLP, self).__init__(*m)

    def forward(self, inputs, batch=None):
        for module in self._modules.values():
            if isinstance(module, (gnn.InstanceNorm)):
                assert batch is not None
                inputs = module(inputs, batch)
            else:
                inputs = module(inputs)
        return inputs


class GIN(GNNBasic):
    def __init__(self, config):

        super().__init__(config)
        self.feat_encoder = GINFeatExtractor(config)
        self.classifier = Classifier(config)
        self.graph_repr = None

    def forward(self, *args, **kwargs):
        out_readout = self.feat_encoder(*args, **kwargs)
        out = self.classifier(out_readout)
        return out


class GINFeatExtractor(GNNBasic):
    def __init__(self, config, **kwargs):
        super(GINFeatExtractor, self).__init__(config, **kwargs)
        num_layer = config['model']['num_layer']
        if config['dataset']['dataset_type'] == 'mol':
            self.encoder = GINMolEncoder(config, **kwargs)
            self.edge_feat = True
        else:
            self.encoder = GINEncoder(config, **kwargs)
            self.edge_feat = False

    def forward(self, *args, **kwargs):

        if self.edge_feat:
            x, edge_index, edge_attr, batch, batch_size = self.arguments_read(*args, edge_feat=self.edge_feat, **kwargs)
            kwargs.pop('batch_size', 'not found')
            out_readout = self.encoder(x, edge_index, edge_attr, batch, batch_size, **kwargs)
        else:
            x, edge_index, batch, batch_size = self.arguments_read(*args, **kwargs)
            kwargs.pop('batch_size', 'not found')
            out_readout = self.encoder(x, edge_index, batch, batch_size, **kwargs)
        return out_readout

    def get_node_repr(self, return_edge=False, *args, **kwargs):
        if self.edge_feat:
            x, edge_index, edge_attr, batch, batch_size = self.arguments_read(*args, edge_feat=self.edge_feat, **kwargs)
            kwargs.pop('batch_size', 'not found')
            if return_edge:
                node_repr, edge_repr = self.encoder.get_node_repr(x, edge_index, edge_attr, batch, batch_size, return_edge, **kwargs)
                return node_repr, edge_repr
            else:
                node_repr = self.encoder.get_node_repr(x, edge_index, edge_attr, batch, batch_size, return_edge, **kwargs)

        else:
            x, edge_index, batch, batch_size = self.arguments_read(*args, **kwargs)
            kwargs.pop('batch_size', 'not found')
            node_repr = self.encoder.get_node_repr(x, edge_index, batch, batch_size, **kwargs)
        return node_repr

class GINFeatExtractor_sub(GNNBasic):
    def __init__(self, config, **kwargs):
        super(GINFeatExtractor_sub, self).__init__(config, **kwargs)
        num_layer = config['model']['num_layer']
        if config['dataset']['dataset_type'] == 'mol':
            self.encoder = GINMolEncoder(config, **kwargs)
            self.edge_feat = True
        else:
            self.encoder = GINEncoder(config, **kwargs)
            self.edge_feat = False

    def forward(self, *args, **kwargs):

        if self.edge_feat:
            x, edge_index, edge_attr, batch, batch_size = self.arguments_read(*args, edge_feat=self.edge_feat, **kwargs)
            kwargs.pop('batch_size', 'not found')
            out_readout = self.encoder(x, edge_index, edge_attr, batch, batch_size, **kwargs)
        else:
            x, edge_index, batch, batch_size = self.arguments_read(*args, **kwargs)
            kwargs.pop('batch_size', 'not found')
            out_readout = self.encoder(x, edge_index, batch, batch_size, **kwargs)
        return out_readout

    def get_node_repr(self, return_edge=False, *args, **kwargs):
        if self.edge_feat:
            x, edge_index, edge_attr, batch, batch_size = self.arguments_read(*args, edge_feat=self.edge_feat, **kwargs)
            kwargs.pop('batch_size', 'not found')
            if return_edge:
                node_repr, edge_repr = self.encoder.get_node_repr(x, edge_index, edge_attr, batch, batch_size, return_edge, **kwargs)
                return node_repr, edge_repr
            else:
                node_repr = self.encoder.get_node_repr(x, edge_index, edge_attr, batch, batch_size, return_edge, **kwargs)

        else:
            x, edge_index, batch, batch_size = self.arguments_read(*args, **kwargs)
            kwargs.pop('batch_size', 'not found')
            node_repr = self.encoder.get_node_repr(x, edge_index, batch, batch_size, **kwargs)
        return node_repr


class GINEncoder(BasicEncoder):
    def __init__(self, config, *args, **kwargs):
        super(GINEncoder, self).__init__(config, *args, **kwargs)
        num_layer = config['model']['num_layer']
        self.without_readout = kwargs.get('without_readout')
        self.y_fuse = kwargs.get('target_fuse_level')
        self.fuse_y_pred = kwargs.get('fuse_y_pred')
        self.fuse_graph_repr = kwargs.get('fuse_graph_repr')
        self.fuse_node_score = kwargs.get('fuse_node_score')
        self.fuse_node_at = kwargs.get('fuse_node_at')
        self.augment_inv = kwargs.get('augment_inv')
        # self.graph_fuser = MLP(
        #     [config['model']['dim_hidden'] * 2, 2 * config['model']['dim_hidden'], config['model']['dim_hidden']],
        #     dropout=config['model']['dropout_rate'], config=config, bn=True) if self.fuse_graph_repr else None
        add_dim = 0
        if self.y_fuse == 'node':
            add_dim += config['model']['dim_hidden']
        if self.fuse_graph_repr:
            add_dim += config['model']['dim_hidden']
        if self.fuse_node_at != 'last':
            self.node_score_proj = nn.Linear(config['model']['dim_hidden'], config['dataset']['dim_node'])
        else:
            self.node_score_proj = None


        self.convs = nn.ModuleList()
        if self.y_fuse == 'node':
            # self.y_emb = nn.Embedding(self.y_classes, config['model']['dim_hidden'])
            self.y_emb = nn.Linear(config['model']['out_shape'], config['model']['dim_hidden'])
            nn.init.xavier_uniform_(self.y_emb.weight.data)
        self.convs.append(gnn.GINConv(nn.Sequential(
            nn.Linear(config['dataset']['dim_node'] + add_dim,
                      2 * config['model']['dim_hidden']),
            nn.BatchNorm1d(2 * config['model']['dim_hidden'], track_running_stats=True),
            nn.ReLU(),
            nn.Linear(2 * config['model']['dim_hidden'], config['model']['dim_hidden']))))
        #     if self.fuse_graph_repr:
        #         self.convs.append(gnn.GINConv(nn.Sequential(
        #             nn.Linear(config['dataset']['dim_node'] + config['model']['dim_hidden'] * 2, 2 * config['model']['dim_hidden']),
        #             nn.BatchNorm1d(2 * config['model']['dim_hidden'], track_running_stats=True),
        #             nn.ReLU(),
        #             nn.Linear(2 * config['model']['dim_hidden'], config['model']['dim_hidden']))))
        #     else:
        #         self.convs.append(gnn.GINConv(nn.Sequential(
        #             nn.Linear(config['dataset']['dim_node'] + config['model']['dim_hidden'], 2 * config['model']['dim_hidden']),
        #             nn.BatchNorm1d(2 * config['model']['dim_hidden'], track_running_stats=True),
        #             nn.ReLU(),
        #             nn.Linear(2 * config['model']['dim_hidden'], config['model']['dim_hidden']))))
        # else:
        #     if self.fuse_graph_repr:
        #         self.convs.append(gnn.GINConv(nn.Sequential(
        #             nn.Linear(config['dataset']['dim_node'] + config['model']['dim_hidden'], 2 * config['model']['dim_hidden']),
        #             nn.BatchNorm1d(2 * config['model']['dim_hidden'], track_running_stats=True),
        #             nn.ReLU(),
        #             nn.Linear(2 * config['model']['dim_hidden'], config['model']['dim_hidden']))))
        #     else:
        #         self.convs.append(gnn.GINConv(nn.Sequential(
        #             nn.Linear(config['dataset']['dim_node'], 2 * config['model']['dim_hidden']),
        #             nn.BatchNorm1d(2 * config['model']['dim_hidden'], track_running_stats=True),
        #             nn.ReLU(),
        #             nn.Linear(2 * config['model']['dim_hidden'], config['model']['dim_hidden']))))

        self.convs = self.convs.extend(
            [
                gnn.GINConv(nn.Sequential(
                    nn.Linear(config['model']['dim_hidden'], 2 * config['model']['dim_hidden']),
                    nn.BatchNorm1d(2 * config['model']['dim_hidden'], track_running_stats=True),
                    nn.ReLU(),
                    nn.Linear(2 * config['model']['dim_hidden'], config['model']['dim_hidden'])))
                for _ in range(num_layer - 1)
            ]
        )

    def forward(self, x, edge_index, batch, batch_size, **kwargs):
        node_repr = self.get_node_repr(x, edge_index, batch, batch_size, **kwargs)
        if self.without_readout or kwargs.get('without_readout'):
            return node_repr
        out_readout = self.readout(node_repr, batch, batch_size)
        return out_readout

    def get_node_repr(self, x, edge_index, batch, batch_size, **kwargs):
        if self.fuse_node_score and self.fuse_node_at != 'last':
            node_score = self.node_score_proj(kwargs.get('node_score'))
            x = torch.mul(x, node_score)
        if self.y_fuse == 'node':
            y = kwargs.get('y_pred')[batch] if self.fuse_y_pred else kwargs.get('data').y[batch].long()
            y_emb = self.y_emb(y)
            x = torch.cat([x, y_emb], dim=1)
        if self.fuse_graph_repr:
            x = torch.cat([x, kwargs.get('graph_repr')[batch]], dim=-1)

        layer_feat = [x]
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            post_conv = batch_norm(conv(layer_feat[-1], edge_index))
            if i != len(self.convs) - 1:
                post_conv = relu(post_conv)
            layer_feat.append(dropout(post_conv))

        if self.fuse_node_score and (self.fuse_node_at == 'last' or self.fuse_node_at == 'both'):
            if not self.augment_inv:
                return torch.mul(layer_feat[-1], kwargs.get('node_score'))
            else:
                return torch.mul(layer_feat[-1], kwargs.get('node_score')), torch.mul(layer_feat[-1], (1-kwargs.get('node_score')))
        else:
            return layer_feat[-1]


class GINEncoder_sub(BasicEncoder):
    def __init__(self, config, *args, **kwargs):
        super(GINEncoder_sub, self).__init__(config, *args, **kwargs)
        num_layer = config['model']['num_layer']
        self.without_readout = kwargs.get('without_readout')
        if kwargs.get('target_fuse_level') == 'node':
            self.y_classes = config['dataset']['y_classes']
            self.fuse_y_pred = kwargs.get('fuse_y_pred')
        else:
            self.y_classes = None
        self.fuse_graph_repr = kwargs.get('fuse_graph_repr')
        self.fuse_node_score = kwargs.get('fuse_node_score')
        self.augment_inv = kwargs.get('augment_inv')
        # self.graph_fuser = MLP(
        #     [config['model']['dim_hidden'] * 2, 2 * config['model']['dim_hidden'], config['model']['dim_hidden']],
        #     dropout=config['model']['dropout_rate'], config=config, bn=True) if self.fuse_graph_repr else None

        self.convs = nn.ModuleList()
        if kwargs.get('without_embed'):
            self.convs.append(gnn.GINConv(nn.Sequential(
                nn.Linear(config['model']['dim_hidden'], 2 * config['model']['dim_hidden']),
                nn.BatchNorm1d(2 * config['model']['dim_hidden'], track_running_stats=True),
                nn.ReLU(),
                nn.Linear(2 * config['model']['dim_hidden'], config['model']['dim_hidden']))))
        elif self.y_classes is not None:
            # self.y_emb = nn.Embedding(self.y_classes, config['model']['dim_hidden'])
            self.y_emb = nn.Linear(1, config['model']['dim_hidden'])
            nn.init.xavier_uniform_(self.y_emb.weight.data)
            if self.fuse_graph_repr:
                self.convs.append(gnn.GINConv(nn.Sequential(
                    nn.Linear(config['dataset']['dim_node'] + config['model']['dim_hidden'] * 2, 2 * config['model']['dim_hidden']),
                    nn.BatchNorm1d(2 * config['model']['dim_hidden'], track_running_stats=True),
                    nn.ReLU(),
                    nn.Linear(2 * config['model']['dim_hidden'], config['model']['dim_hidden']))))
            else:
                self.convs.append(gnn.GINConv(nn.Sequential(
                    nn.Linear(config['dataset']['dim_node'] + config['model']['dim_hidden'], 2 * config['model']['dim_hidden']),
                    nn.BatchNorm1d(2 * config['model']['dim_hidden'], track_running_stats=True),
                    nn.ReLU(),
                    nn.Linear(2 * config['model']['dim_hidden'], config['model']['dim_hidden']))))
        else:
            self.convs.append(gnn.GINConv(nn.Sequential(
                nn.Linear(config['dataset']['dim_node'], 2 * config['model']['dim_hidden']),
                nn.BatchNorm1d(2 * config['model']['dim_hidden'], track_running_stats=True),
                nn.ReLU(),
                nn.Linear(2 * config['model']['dim_hidden'], config['model']['dim_hidden']))))

        self.convs = self.convs.extend(
            [
                gnn.GINConv(nn.Sequential(
                    nn.Linear(config['model']['dim_hidden'], 2 * config['model']['dim_hidden']),
                    nn.BatchNorm1d(2 * config['model']['dim_hidden'], track_running_stats=True),
                    nn.ReLU(),
                    nn.Linear(2 * config['model']['dim_hidden'], config['model']['dim_hidden'])))
                for _ in range(num_layer - 1)
            ]
        )

    def forward(self, x, edge_index, batch, batch_size, **kwargs):
        node_repr = self.get_node_repr(x, edge_index, batch, batch_size, **kwargs)
        if self.without_readout or kwargs.get('without_readout'):
            return node_repr
        out_readout = self.readout(node_repr, batch, batch_size)
        return out_readout

    def get_node_repr(self, x, edge_index, batch, batch_size, **kwargs):
        if self.y_classes is not None:
            y = kwargs.get('y_pred') if self.fuse_y_pred else kwargs.get('data').y[batch].long()
            y_emb = self.y_emb(y)
            x = torch.cat([x, y_emb], dim=1)
        if self.fuse_graph_repr:
            x = torch.cat([x, kwargs.get('graph_repr')[batch]], dim=-1)
        if self.fuse_node_score:
            x = torch.mul(x, kwargs.get('node_score'))
        layer_feat = [x]
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            post_conv = batch_norm(conv(layer_feat[-1], edge_index))
            if i != len(self.convs) - 1:
                post_conv = relu(post_conv)
            layer_feat.append(dropout(post_conv))
        return layer_feat[-1]


class GINMolEncoder(BasicEncoder):
    def __init__(self, config, **kwargs):
        super(GINMolEncoder, self).__init__(config, **kwargs)
        self.without_readout = kwargs.get('without_readout')
        self.num_layer = config['model']['num_layer']
        self.y_fuse = kwargs.get('target_fuse_level')
        self.fuse_y_pred = kwargs.get('fuse_y_pred')
        self.without_readout = kwargs.get('without_readout')
        self.fuse_graph_repr = kwargs.get('fuse_graph_repr')
        self.fuse_node_score = kwargs.get('fuse_node_score')
        self.fuse_node_at = kwargs.get('fuse_node_at')
        self.augment_inv = kwargs.get('augment_inv')
        self.bond_conv = config['model']['bond_encoder_conv']

        self.graph_fuser = MLP([config['model']['dim_hidden'] * 2, 2 * config['model']['dim_hidden'], config['model']['dim_hidden']],
            dropout=config['model']['dropout_rate'], config=config, bn=True) if self.fuse_graph_repr else None
        if self.y_fuse:
            self.y_emb = nn.Linear(config['model']['out_shape'], config['model']['dim_hidden'])

        if kwargs.get('without_embed'):
            self.atom_encoder = Identity()
        else:
            self.atom_encoder = AtomEncoder(config['model']['dim_hidden'], config)
        if not self.bond_conv:
            self.bond_encoder = BondEncoder(config['model']['dim_hidden'], config)
        else:
            self.bond_encoder = nn.ModuleList(
                [BondEncoder(config['model']['dim_hidden'], config)
                 for _ in range(self.num_layer)]
            )
        self.convs = nn.ModuleList(
            [
                GINEConv(nn.Sequential(
                    nn.Linear(config['model']['dim_hidden'], 2 * config['model']['dim_hidden']),
                    nn.BatchNorm1d(2 * config['model']['dim_hidden']), nn.ReLU(),
                    nn.Linear(2 * config['model']['dim_hidden'], config['model']['dim_hidden'])), config)
                for _ in range(self.num_layer)
            ]
        )

    def forward(self, x, edge_index, edge_attr, batch, batch_size, **kwargs):
        node_repr = self.get_node_repr(x, edge_index, edge_attr, batch, batch_size, return_edge=False, **kwargs)

        if self.without_readout or kwargs.get('without_readout'):
            return node_repr
        # torch.use_deterministic_algorithms(False)
        out_readout = self.readout(node_repr, batch, batch_size)
        # torch.use_deterministic_algorithms(True)
        return out_readout

    def get_node_repr(self, x, edge_index, edge_attr, batch, batch_size, return_edge, **kwargs):
        x = self.atom_encoder(x)
        if self.y_fuse == 'node':
            y = kwargs['y_pred'][batch] if self.fuse_y_pred else kwargs.get('data').y[batch].long()
            x = x + self.y_emb(y)
            # x = torch.cat([x, y], dim=1)
        if self.graph_fuser:
            x = torch.cat([x, kwargs.get('graph_repr')[batch]], dim=-1)
            x = self.graph_fuser(x)
        if self.fuse_node_score and self.fuse_node_at != 'last':
            x = torch.mul(x, kwargs.get('node_score'))
        layer_feat = [x]
        if not self.bond_conv:
            edge_attr = self.bond_encoder(edge_attr)
            for i, (conv, batch_norm, relu, dropout) in enumerate(
                    zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
                if i < self.num_layer - 1:
                    post_conv = conv(layer_feat[-1], edge_index, edge_attr)
                    post_conv = batch_norm(post_conv)
                    post_conv = relu(post_conv)
                else:
                    if return_edge and self.bond_encoder is None:
                        post_conv, edge_repr = conv(layer_feat[-1], edge_index, edge_attr, return_edge)
                    else:
                        post_conv = conv(layer_feat[-1], edge_index, edge_attr)
                    post_conv = batch_norm(post_conv)
                layer_feat.append(dropout(post_conv))
        else:
            for i, (conv, bond_encoder, batch_norm, relu, dropout) in enumerate(
                    zip(self.convs, self.bond_encoder, self.batch_norms, self.relus, self.dropouts)):
                edge_feat = bond_encoder(edge_attr)
                # adj_t = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_feat).t()
                if i < self.num_layer - 1:
                    # post_conv = conv(layer_feat[-1], adj_t)
                    post_conv = conv(layer_feat[-1], edge_index, edge_feat)
                    # post_conv = conv(layer_feat[-1], edge_index, edge_attr)
                    post_conv = batch_norm(post_conv)
                    post_conv = relu(post_conv)
                else:
                    # adj_t = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_attr).t()
                    # post_conv, edge_repr = conv(layer_feat[-1], adj_t, return_edge)
                    # post_conv, edge_repr = conv(layer_feat[-1], edge_index, edge_attr, return_edge)

                    # post_conv = conv(layer_feat[-1], adj_t)
                    post_conv = conv(layer_feat[-1], edge_index, edge_feat)
                    # post_conv = conv(layer_feat[-1], edge_index, edge_attr)
                    post_conv = batch_norm(post_conv)
                layer_feat.append(dropout(post_conv))
        if return_edge and self.bond_encoder is not None:
            return layer_feat[-1], edge_attr
        elif return_edge:
            return layer_feat[-1], edge_repr

        if self.fuse_node_score and (self.fuse_node_at == 'last' or self.fuse_node_at == 'both'):
            if not self.augment_inv:
                return torch.mul(layer_feat[-1], kwargs.get('node_score'))
            else:
                return torch.mul(layer_feat[-1], kwargs.get('node_score')), torch.mul(layer_feat[-1], (1-kwargs.get('node_score')))
        else:
            return layer_feat[-1]


class GINMolEncoder_sub(BasicEncoder):
    def __init__(self, config, **kwargs):
        super(GINMolEncoder_sub, self).__init__(config, **kwargs)
        self.without_readout = kwargs.get('without_readout')
        self.num_layer = config['model']['num_layer']
        self.y_fuse = kwargs.get('target_fuse_level')
        self.fuse_y_pred = kwargs.get('fuse_y_pred')
        self.without_readout = kwargs.get('without_readout')
        self.fuse_graph_repr = kwargs.get('fuse_graph_repr')
        self.fuse_node_score = kwargs.get('fuse_node_score')
        self.bond_conv = config['model']['bond_encoder_conv']

        self.graph_fuser = MLP([config['model']['dim_hidden'] * 2, 2 * config['model']['dim_hidden'], config['model']['dim_hidden']],
            dropout=config['model']['dropout_rate'], config=config, bn=True) if self.fuse_graph_repr else None
        if self.y_fuse:
            self.y_emb = nn.Linear(1, config['model']['dim_hidden'])

        if kwargs.get('without_embed'):
            self.atom_encoder = Identity()
        else:
            self.atom_encoder = AtomEncoder(config['model']['dim_hidden'], config)
        if not self.bond_conv:
            self.bond_encoder = BondEncoder(config['model']['dim_hidden'], config)
        else:
            self.bond_encoder = nn.ModuleList(
                [BondEncoder(config['model']['dim_hidden'], config)
                 for _ in range(self.num_layer)]
            )
        self.convs = nn.ModuleList(
            [
                GINEConv(nn.Sequential(
                    nn.Linear(config['model']['dim_hidden'], 2 * config['model']['dim_hidden']),
                    nn.BatchNorm1d(2 * config['model']['dim_hidden']), nn.ReLU(),
                    nn.Linear(2 * config['model']['dim_hidden'], config['model']['dim_hidden'])), config)
                for _ in range(self.num_layer)
            ]
        )

    def forward(self, x, edge_index, edge_attr, batch, batch_size, **kwargs):
        node_repr = self.get_node_repr(x, edge_index, edge_attr, batch, batch_size, return_edge=False, **kwargs)

        if self.without_readout or kwargs.get('without_readout'):
            return node_repr
        # torch.use_deterministic_algorithms(False)
        out_readout = self.readout(node_repr, batch, batch_size)
        # torch.use_deterministic_algorithms(True)
        return out_readout

    def get_node_repr(self, x, edge_index, edge_attr, batch, batch_size, return_edge, **kwargs):
        input_dict = kwargs[kwargs['stage']]
        # if input_dict.get('')

        x = self.atom_encoder(x)
        if self.y_fuse == 'node':
            y = kwargs['y_pred'][batch] if self.fuse_y_pred else kwargs.get('data').y[batch].long()
            x = x + self.y_emb(y)
            # x = torch.cat([x, y], dim=1)
        if self.graph_fuser:
            x = torch.cat([x, kwargs.get('graph_repr')[batch]], dim=-1)
            x = self.graph_fuser(x)
        if self.fuse_node_score:
            x = torch.mul(x, kwargs.get('node_score'))
        layer_feat = [x]
        if not self.bond_conv:
            edge_attr = self.bond_encoder(edge_attr)
            for i, (conv, batch_norm, relu, dropout) in enumerate(
                    zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
                if i < self.num_layer - 1:
                    post_conv = conv(layer_feat[-1], edge_index, edge_attr)
                    post_conv = batch_norm(post_conv)
                    post_conv = relu(post_conv)
                else:
                    if return_edge and self.bond_encoder is None:
                        post_conv, edge_repr = conv(layer_feat[-1], edge_index, edge_attr, return_edge)
                    else:
                        post_conv = conv(layer_feat[-1], edge_index, edge_attr)
                    post_conv = batch_norm(post_conv)
                layer_feat.append(dropout(post_conv))
        else:
            for i, (conv, bond_encoder, batch_norm, relu, dropout) in enumerate(
                    zip(self.convs, self.bond_encoder, self.batch_norms, self.relus, self.dropouts)):
                edge_feat = bond_encoder(edge_attr)
                # adj_t = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_feat).t()
                if i < self.num_layer - 1:
                    # post_conv = conv(layer_feat[-1], adj_t)
                    post_conv = conv(layer_feat[-1], edge_index, edge_feat)
                    # post_conv = conv(layer_feat[-1], edge_index, edge_attr)
                    post_conv = batch_norm(post_conv)
                    post_conv = relu(post_conv)
                else:
                    # adj_t = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_attr).t()
                    # post_conv, edge_repr = conv(layer_feat[-1], adj_t, return_edge)
                    # post_conv, edge_repr = conv(layer_feat[-1], edge_index, edge_attr, return_edge)

                    # post_conv = conv(layer_feat[-1], adj_t)
                    post_conv = conv(layer_feat[-1], edge_index, edge_feat)
                    # post_conv = conv(layer_feat[-1], edge_index, edge_attr)
                    post_conv = batch_norm(post_conv)
                layer_feat.append(dropout(post_conv))
        if return_edge and self.bond_encoder is not None:
            return layer_feat[-1], edge_attr
        elif return_edge:
            return layer_feat[-1], edge_repr
        return layer_feat[-1]


class GINConv(gnn.MessagePassing):
    def __init__(self, network):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr="add")

        self.mlp = network
        self.eps = torch.nn.Parameter(torch.Tensor([0]))

        # self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, x, edge_index, edge_attr):
        # edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_attr))

        return out

    def message(self, x_j, edge_attr):
        return nn.functional.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

class GINEConv(gnn.MessagePassing):
    def __init__(self, nn, config, eps=0., train_eps=False, edge_dim=None, return_edge=False, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        self.nn = nn
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

        if hasattr(self.nn[0], 'in_features'):
            in_channels = self.nn[0].in_features
        else:
            in_channels = self.nn[0].in_channels
        # self.bone_encoder = BondEncoder(in_channels, config) if config['model']['bond_encoder_conv'] else None
        self.lin = None
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)
        self.eps.data.fill_(self.initial_eps)
        if self.lin is not None:
            self.lin.reset_parameters()

    def forward(self, x, edge_index, edge_attr=None, size=None, return_edge=False):
        # if self.bone_encoder and edge_attr is not None:
        #     edge_attr = self.bone_encoder(edge_attr)
        if isinstance(x, torch.Tensor):
            x: OptPairTensor = (x, x)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr, size=size)

        x_r = x[1]
        if x_r is not None:
            out += (1 + self.eps) * x_r
        if return_edge and self.bone_encoder is not None:
            return self.nn(out), edge_attr
        else:
            return self.nn(out)

    def message(self, x_j, edge_attr):
        if edge_attr is not None:
            if self.lin is None and x_j.size(-1) != edge_attr.size(-1):
                raise ValueError("Node and edge feature dimensionalities do not "
                                 "match. Consider setting the 'edge_dim' "
                                 "attribute of 'GINEConv'")

            if self.lin is not None:
                edge_attr = self.lin(edge_attr)
            m = x_j + edge_attr
        else:
            m = x_j
        return m.relu()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'

