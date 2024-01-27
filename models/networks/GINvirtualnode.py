import torch
import torch.nn as nn
from torch_sparse import SparseTensor

from .BaseGNN import GNNBasic
from .Classifiers import Classifier
from .GINs import GINEncoder, GINMolEncoder, GINFeatExtractor, GINEncoder_sub, GINMolEncoder_sub, GINFeatExtractor_sub
from .Pooling import GlobalAddPool


class vGIN(GNNBasic):
    def __init__(self, config):
        super(vGIN, self).__init__(config)
        self.feat_encoder = vGINFeatExtractor(config)
        self.classifier = Classifier(config)
        self.graph_repr = None

    def forward(self, *args, **kwargs):
        out_readout = self.feat_encoder(*args, **kwargs)
        out = self.classifier(out_readout)
        return out


class vGINFeatExtractor(GINFeatExtractor):
    def __init__(self, config, **kwargs):
        super(vGINFeatExtractor, self).__init__(config, **kwargs)
        num_layer = config['model']['num_layer']
        if config['dataset']['dataset_type'] == 'mol':
            self.encoder = vGINMolEncoder(config, **kwargs)
            self.edge_feat = True
        else:
            self.encoder = vGINEncoder(config, **kwargs)
            self.edge_feat = False

class vGINFeatExtractor_sub(GINFeatExtractor_sub):
    def __init__(self, config, **kwargs):
        super(vGINFeatExtractor_sub, self).__init__(config, **kwargs)
        num_layer = config['model']['num_layer']
        if config['dataset']['dataset_type'] == 'mol':
            self.encoder = vGINMolEncoder(config, **kwargs)
            self.edge_feat = True
        else:
            self.encoder = vGINEncoder(config, **kwargs)
            self.edge_feat = False


class VirtualNodeEncoder(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super(VirtualNodeEncoder, self).__init__()
        self.virtual_node_embedding = nn.Embedding(1, config['model']['dim_hidden'])
        self.virtual_mlp = nn.Sequential(*(
                [nn.Linear(config['model']['dim_hidden'], 2 * config['model']['dim_hidden']),
                 nn.BatchNorm1d(2 * config['model']['dim_hidden']), nn.ReLU()] +
                [nn.Linear(2 * config['model']['dim_hidden'], config['model']['dim_hidden']),
                 nn.BatchNorm1d(config['model']['dim_hidden']), nn.ReLU(),
                 nn.Dropout(config['model']['dropout_rate'])]
        ))
        self.virtual_pool = GlobalAddPool()


class vGINEncoder(GINEncoder, VirtualNodeEncoder):
    def __init__(self, config, **kwargs):
        super(vGINEncoder, self).__init__(config, **kwargs)
        self.config = config
        self.without_readout = kwargs.get('without_readout')

    def forward(self, x, edge_index, batch, batch_size, **kwargs):
        node_repr = self.get_node_repr(x, edge_index, batch, batch_size, **kwargs)

        if self.without_readout or kwargs.get('without_readout'):
            return node_repr
        out_readout = self.readout(node_repr, batch, batch_size)
        return out_readout

    def get_node_repr(self, x, edge_index, batch, batch_size, **kwargs):
        virtual_node_feat = [self.virtual_node_embedding(
            torch.zeros(batch[-1].item() + 1, device=self.config['device'], dtype=torch.long))]
        if self.fuse_node_score and self.fuse_node_at != 'last':
            node_score = self.node_score_proj(kwargs.get('node_score'))
            x = torch.mul(x, node_score)
        if self.y_fuse == 'node':
            y = kwargs['y_pred'][batch] if self.fuse_y_pred else kwargs.get('data').y[batch].long()
            x = torch.cat([x, self.y_emb(y)], dim=-1)
        # if self.y_classes is not None:
        #     y = kwargs.get('y_pred') if self.fuse_y_pred else kwargs.get('data').y[batch].long()
        #     y_emb = self.y_emb(y)
        #     x = torch.cat([x, y_emb], dim=1)
        if self.fuse_graph_repr:
            x = torch.cat([x, kwargs.get('graph_repr')[batch]], dim=-1)

        layer_feat = [x]
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            # --- Add global info ---
            if i > 0:
                post_conv = layer_feat[-1] + virtual_node_feat[-1][batch]
            else:
                post_conv = layer_feat[-1]
            post_conv = batch_norm(conv(post_conv, edge_index))
            if i < len(self.convs) - 1:
                post_conv = relu(post_conv)
            layer_feat.append(dropout(post_conv))
            # --- update global info ---
            if 0 < i < len(self.convs) - 1:
                virtual_node_feat.append(
                    self.virtual_mlp(self.virtual_pool(layer_feat[-1], batch, batch_size) + virtual_node_feat[-1]))

        if self.fuse_node_score and (self.fuse_node_at == 'last' or self.fuse_node_at == 'both'):
            if not self.augment_inv:
                return torch.mul(layer_feat[-1], kwargs.get('node_score'))
            else:
                return torch.mul(layer_feat[-1], kwargs.get('node_score')), torch.mul(layer_feat[-1], (1-kwargs.get('node_score')))
        else:
            return layer_feat[-1]


class vGINEncoder_sub(GINEncoder, VirtualNodeEncoder):
    def __init__(self, config, **kwargs):
        super(vGINEncoder, self).__init__(config, **kwargs)
        self.config = config
        self.without_readout = kwargs.get('without_readout')

    def forward(self, x, edge_index, batch, batch_size, **kwargs):
        node_repr = self.get_node_repr(x, edge_index, batch, batch_size, **kwargs)

        if self.without_readout or kwargs.get('without_readout'):
            return node_repr
        out_readout = self.readout(node_repr, batch, batch_size)
        return out_readout

    def get_node_repr(self, x, edge_index, batch, batch_size, **kwargs):
        virtual_node_feat = [self.virtual_node_embedding(
            torch.zeros(batch[-1].item() + 1, device=self.config['device'], dtype=torch.long))]
        if self.y_fuse == 'node':
            y = kwargs.get('y_pred')[batch] if self.fuse_y_pred else kwargs.get('data').y[batch].long()
            y_emb = self.y_emb(y)
            x = torch.cat([x, y_emb], dim=1)
        if self.fuse_graph_repr:
            x = torch.cat([x, kwargs.get('graph_repr')[batch]], dim=-1)
        if self.fuse_node_score:
            x = torch.mul(x, kwargs.get('node_score'))
        layer_feat = [x]
        for i, (conv, batch_norm, relu, dropout) in enumerate(
                zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
            # --- Add global info ---
            if i > 0:
                post_conv = layer_feat[-1] + virtual_node_feat[-1][batch]
            else:
                post_conv = layer_feat[-1]
            post_conv = batch_norm(conv(post_conv, edge_index))
            if i < len(self.convs) - 1:
                post_conv = relu(post_conv)
            layer_feat.append(dropout(post_conv))
            # --- update global info ---
            if 0 < i < len(self.convs) - 1:
                virtual_node_feat.append(
                    self.virtual_mlp(self.virtual_pool(layer_feat[-1], batch, batch_size) + virtual_node_feat[-1]))
        return layer_feat[-1]



class vGINMolEncoder(GINMolEncoder, VirtualNodeEncoder):
    def __init__(self, config, **kwargs):
        super(vGINMolEncoder, self).__init__(config, **kwargs)
        self.config = config

    def forward(self, x, edge_index, edge_attr, batch, batch_size, **kwargs):
        node_repr = self.get_node_repr(x, edge_index, edge_attr, batch, batch_size, return_edge=False, **kwargs)

        if self.without_readout or kwargs.get('without_readout'):
            return node_repr

        # torch.use_deterministic_algorithms(False)
        out_readout = self.readout(node_repr, batch, batch_size)
        # torch.use_deterministic_algorithms(True)
        return out_readout

    def get_node_repr(self, x, edge_index, edge_attr, batch, batch_size, return_edge, **kwargs):
        virtual_node_feat = [self.virtual_node_embedding(
            torch.zeros(batch[-1].item() + 1, device=self.config['device'], dtype=torch.long))]

        # for i, sub_x in enumerate(x):
        #     print(i)
        #     self.atom_encoder(sub_x[None, :])
        x = self.atom_encoder(x)
        # adj = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_attr).t()
        if self.y_fuse == 'node':
            y = kwargs['y_pred'][batch] if self.fuse_y_pred else kwargs.get('data').y[batch].long()
            x = x + self.y_emb(y)
            # x = torch.cat([x, y], dim=1)

        if self.fuse_graph_repr and self.fuse_node_at != 'last':
            x = torch.cat([x, kwargs.get('graph_repr')[batch]], dim=-1)
            x = self.graph_fuser(x)
        if self.fuse_node_score:
            x = torch.mul(x, kwargs.get('node_score'))
        layer_feat = [x]
        if not self.bond_conv:
            edge_feat = self.bond_encoder(edge_attr)
            for i, (conv, batch_norm, relu, dropout) in enumerate(
                    zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
                # --- Add global info ---
                post_conv = layer_feat[-1] + virtual_node_feat[-1][batch]
                if i < self.num_layer - 1:
                    # post_conv = conv(post_conv, adj.t())
                    post_conv = conv(post_conv, edge_index, edge_feat)
                    post_conv = dropout(relu(batch_norm(post_conv)))
                    layer_feat.append(post_conv)
                    virtual_node_feat.append(
                        self.virtual_mlp(self.virtual_pool(layer_feat[-1], batch, batch_size) + virtual_node_feat[-1]))
                else:
                    # if return_edge and self.bond_encoder is None:
                    #     post_conv, edge_repr = conv(post_conv, edge_index, edge_attr, return_edge=return_edge)
                    # else:
                    post_conv = conv(post_conv, edge_index, edge_attr)
                    post_conv = batch_norm(post_conv)
                    layer_feat.append(dropout(post_conv))
        else:
            for i, (conv, bond_encoder, batch_norm, relu, dropout) in enumerate(
                    zip(self.convs, self.bond_encoder, self.batch_norms, self.relus, self.dropouts)):
                # --- Add global info ---
                post_conv = layer_feat[-1] + virtual_node_feat[-1][batch]
                edge_feat = bond_encoder(edge_attr)
                # adj_t = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_feat).t()
                if i < self.num_layer - 1:
                    # post_conv = conv(post_conv, adj_t)
                    post_conv = conv(post_conv, edge_index, edge_feat)
                    # post_conv = conv(post_conv, edge_index, edge_attr)
                    post_conv = dropout(relu(batch_norm(post_conv)))
                    layer_feat.append(post_conv)

                    # torch.use_deterministic_algorithms(False)
                    virtual_node_feat.append(
                        self.virtual_mlp(self.virtual_pool(layer_feat[-1], batch, batch_size) + virtual_node_feat[-1]))

                    # torch.use_deterministic_algorithms(True)
                else:
                    # if return_edge and self.bond_encoder is None:
                    #     post_conv, edge_repr = conv(post_conv, edge_index, edge_attr, return_edge=return_edge)
                    # else:
                    # post_conv = conv(post_conv, adj_t)
                    post_conv = conv(post_conv, edge_index, edge_feat)
                    # post_conv = conv(post_conv, edge_index, edge_attr)
                    post_conv = batch_norm(post_conv)
                    layer_feat.append(dropout(post_conv))

        if self.fuse_node_score and (self.fuse_node_at == 'last' or self.fuse_node_at == 'both'):
            if not self.augment_inv:
                return torch.mul(layer_feat[-1], kwargs.get('node_score'))
            else:
                return torch.mul(layer_feat[-1], kwargs.get('node_score')), torch.mul(layer_feat[-1], (1-kwargs.get('node_score')))
        if return_edge and self.bond_encoder is not None:
            return layer_feat[-1], edge_attr
        # elif return_edge:
        #     return layer_feat[-1], edge_repr
        return layer_feat[-1]


class vGINMolEncoder_sub(GINMolEncoder, VirtualNodeEncoder):
    def __init__(self, config, **kwargs):
        super(vGINMolEncoder_sub, self).__init__(config, **kwargs)
        self.config = config

    def forward(self, x, edge_index, edge_attr, batch, batch_size, **kwargs):
        node_repr = self.get_node_repr(x, edge_index, edge_attr, batch, batch_size, return_edge=False, **kwargs)

        if self.without_readout or kwargs.get('without_readout'):
            return node_repr

        # torch.use_deterministic_algorithms(False)
        out_readout = self.readout(node_repr, batch, batch_size)
        # torch.use_deterministic_algorithms(True)
        return out_readout

    def get_node_repr(self, x, edge_index, edge_attr, batch, batch_size, return_edge, **kwargs):
        virtual_node_feat = [self.virtual_node_embedding(
            torch.zeros(batch[-1].item() + 1, device=self.config['device'], dtype=torch.long))]

        # for i, sub_x in enumerate(x):
        #     print(i)
        #     self.atom_encoder(sub_x[None, :])
        x = self.atom_encoder(x)
        # adj = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_attr).t()
        if self.y_fuse == 'node':
            y = kwargs['y_pred'][batch] if self.fuse_y_pred else kwargs.get('data').y[batch].long()
            x = x + self.y_emb(y)
            # x = torch.cat([x, y], dim=1)

        if self.fuse_graph_repr:
            x = torch.cat([x, kwargs.get('graph_repr')[batch]], dim=-1)
            x = self.graph_fuser(x)
        if self.fuse_node_score and self.fuse_node_at != 'last':
            x = torch.mul(x, kwargs.get('node_score'))
        layer_feat = [x]
        if not self.bond_conv:
            edge_feat = self.bond_encoder(edge_attr)
            for i, (conv, batch_norm, relu, dropout) in enumerate(
                    zip(self.convs, self.batch_norms, self.relus, self.dropouts)):
                # --- Add global info ---
                post_conv = layer_feat[-1] + virtual_node_feat[-1][batch]
                if i < self.num_layer - 1:
                    # post_conv = conv(post_conv, adj.t())
                    post_conv = conv(post_conv, edge_index, edge_feat)
                    post_conv = dropout(relu(batch_norm(post_conv)))
                    layer_feat.append(post_conv)
                    virtual_node_feat.append(
                        self.virtual_mlp(self.virtual_pool(layer_feat[-1], batch, batch_size) + virtual_node_feat[-1]))
                else:
                    # if return_edge and self.bond_encoder is None:
                    #     post_conv, edge_repr = conv(post_conv, edge_index, edge_attr, return_edge=return_edge)
                    # else:
                    post_conv = conv(post_conv, edge_index, edge_attr)
                    post_conv = batch_norm(post_conv)
                    layer_feat.append(dropout(post_conv))
        else:
            for i, (conv, bond_encoder, batch_norm, relu, dropout) in enumerate(
                    zip(self.convs, self.bond_encoder, self.batch_norms, self.relus, self.dropouts)):
                # --- Add global info ---
                post_conv = layer_feat[-1] + virtual_node_feat[-1][batch]
                edge_feat = bond_encoder(edge_attr)
                # adj_t = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_feat).t()
                if i < self.num_layer - 1:
                    # post_conv = conv(post_conv, adj_t)
                    post_conv = conv(post_conv, edge_index, edge_feat)
                    # post_conv = conv(post_conv, edge_index, edge_attr)
                    post_conv = dropout(relu(batch_norm(post_conv)))
                    layer_feat.append(post_conv)

                    # torch.use_deterministic_algorithms(False)
                    virtual_node_feat.append(
                        self.virtual_mlp(self.virtual_pool(layer_feat[-1], batch, batch_size) + virtual_node_feat[-1]))

                    # torch.use_deterministic_algorithms(True)
                else:
                    # if return_edge and self.bond_encoder is None:
                    #     post_conv, edge_repr = conv(post_conv, edge_index, edge_attr, return_edge=return_edge)
                    # else:
                    # post_conv = conv(post_conv, adj_t)
                    post_conv = conv(post_conv, edge_index, edge_feat)
                    # post_conv = conv(post_conv, edge_index, edge_attr)
                    post_conv = batch_norm(post_conv)
                    layer_feat.append(dropout(post_conv))

        if return_edge and self.bond_encoder is not None:
            return layer_feat[-1], edge_attr

        if self.fuse_node_score and (self.fuse_node_at == 'last' or self.fuse_node_at == 'both'):
            return torch.mul(layer_feat[-1], kwargs.get('node_score'))
        # elif return_edge:
        #     return layer_feat[-1], edge_repr
        else:
            return layer_feat[-1]
