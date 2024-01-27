import torch
import torch.nn as nn
import torch_geometric.nn as gnn

from .BaseGNN import GNNBasic
from .GINs import GINFeatExtractor
from .GINvirtualnode import vGINFeatExtractor
from utils import set_seed


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


class PseudoLabelClassifier(nn.Module):
    def __init__(self, config):
        super(PseudoLabelClassifier, self).__init__()
        hidden_size = config['model']['dim_hidden']
        dropout_p = config['model']['dropout_rate']
        fg_kwargs = {'target_fuse_level': None,
                     'fuse_y_pred': False}
        self.relu_out = config['model']['relu_out_e']
        if config['model']['gnn_kernel_1'] == 'vGIN':
            self.feat_extractor = vGINFeatExtractor(config, **fg_kwargs)
        elif config['model']['gnn_kernel_1'] == 'GIN':
            self.feat_extractor = GINFeatExtractor(config, **fg_kwargs)

        self.out_mlp = MLP([hidden_size, hidden_size * 2, hidden_size, config['model']['out_shape']], dropout=dropout_p,
                                     config=config, bn=True)
        self.out_shape = config['model']['out_shape']

    def forward(self, *args, **kwargs):
        # data = kwargs.get('data')
        graph_repr = self.feat_extractor(*args, **kwargs)
        y_prob = self.out_mlp(graph_repr)
        if self.relu_out:
            if self.out_shape == 1:
                y_prob = torch.relu(y_prob)
            else:
                y_prob = torch.softmax(y_prob, dim=-1)
        y_pred = y_prob
        return y_pred, y_prob


class EnvGenerator(nn.Module):
    def __init__(self, config, **kwargs):
        super(EnvGenerator, self).__init__()

        eg_kwargs = {'target_fuse_level': 'node',
                     'fuse_y_pred': True}
        if config['model']['gnn_kernel_2'] == 'vGIN':
            self.feat_extractor = vGINFeatExtractor(config, **eg_kwargs)
        elif config['model']['gnn_kernel_2'] == 'GIN':
            self.feat_extractor = GINFeatExtractor(config, **eg_kwargs)
        self.grl_env = config['model']['grl_env']
        self.alpha = config['train']['alpha'] * config['ood']['env_f']

        self.label_fuser = MLP(
            [config['model']['dim_hidden'] + config['model']['out_shape'], 2 * config['model']['dim_hidden'], config['model']['dim_hidden']],
            dropout=config['model']['dropout_rate'], config=config, bn=True)
    def forward(self, *args, **kwargs):
        if not self.grl_env:
            graph_repr = self.feat_extractor(*args, **kwargs)
            y = kwargs.get('y_pred')
            graph_repr = torch.sigmoid(self.label_fuser(torch.cat([graph_repr, y], dim=-1)))
        else:
            node_repr = self.feat_extractor.get_node_repr(*args, **kwargs)
            y = kwargs.get('y_pred')
            node_repr = GradientReverseLayerF.apply(node_repr, self.alpha)
            graph_repr = self.feat_extractor.encoder.readout(node_repr, kwargs['data'].batch)
            graph_repr = torch.sigmoid(self.label_fuser(torch.cat([graph_repr, y], dim=-1)))
        return graph_repr


class GraphRationaleExtractor(nn.Module):
    def __init__(self, config, **kwargs):
        super(GraphRationaleExtractor, self).__init__()
        hidden_size = config['model']['dim_hidden']
        dropout_p = config['model']['dropout_rate']
        self.augment_inv = config['model']['augment_inv']

        ig_kwargs = {'target_fuse_level': 'node',
                     'fuse_y_pred': True,
                     'fuse_graph_repr': True}

        self.grl_inv = config['model']['grl_inv']
        self.alpha = config['train']['alpha'] * config['ood']['inv_f']

        if config['model']['gnn_kernel_3'] == 'vGIN':
            self.graph_encoder = vGINFeatExtractor(config, **ig_kwargs)
        elif config['model']['gnn_kernel_3'] == 'GIN':
            self.graph_encoder = GINFeatExtractor(config, **ig_kwargs)

        self.label_fuser = MLP([hidden_size + config['model']['out_shape'], 2 * hidden_size, hidden_size],
                               dropout=dropout_p, config=config, bn=True)

    def forward(self, *args, **kwargs):
        if not self.grl_inv:
            node_repr = self.graph_encoder.get_node_repr(return_edge=False, **kwargs)
            y = kwargs.get('y_pred')[kwargs['data'].batch]
            node_repr = self.label_fuser(torch.cat([node_repr, y], dim=-1))
            node_score = torch.sigmoid(node_repr)
        else:
            node_repr = self.graph_encoder.get_node_repr(return_edge=False, **kwargs)
            # if self.training:
            #     node_repr = (1 - GradientReverseLayerF.apply(node_repr, self.alpha))
            node_repr = (1 - GradientReverseLayerF.apply(node_repr, self.alpha))
            y = kwargs.get('y_pred')[kwargs['data'].batch]
            node_repr = self.label_fuser(torch.cat([node_repr, y], dim=-1))
            node_score = torch.sigmoid(node_repr)
        return node_score


class FinalClassifier(nn.Module):
    def __init__(self, config, **kwargs):
        super(FinalClassifier, self).__init__()
        hidden_size = config['model']['dim_hidden']
        dropout_p = config['model']['dropout_rate']
        self.augment_inv = config['model']['augment_inv']
        self.augment_inv = False

        ega_kwargs = {'target_fuse_level': 'node',
                      'fuse_y_pred': True,
                      'fuse_graph_repr': True,
                      'fuse_node_score': True,
                      'fuse_node_at': config['model']['fuse_node_at'],
                      'augment_inv': self.augment_inv}

        if config['model']['gnn_kernel_4'] == 'vGIN':
            self.graph_encoder = vGINFeatExtractor(config, **ega_kwargs)
        elif config['model']['gnn_kernel_4'] == 'GIN':
            self.graph_encoder = GINFeatExtractor(config, **ega_kwargs)

        self.label_predictor = MLP([hidden_size, 2 * hidden_size, hidden_size, config['model']['out_shape']],
                                   dropout=dropout_p, config=config, bn=True)
        self.out_shape = config['model']['out_shape']

    def forward(self, *args, **kwargs):
        if self.augment_inv:
            node_repr, other_node_repr = self.graph_encoder.get_node_repr(*args, **kwargs)
            graph_repr = self.graph_encoder.encoder.readout(node_repr, kwargs['data'].batch)
            other_graph_repr = self.graph_encoder.encoder.readout(other_node_repr, kwargs['data'].batch)
            y_ega_prob = self.label_predictor(graph_repr)
            y_other_prob = self.label_predictor(other_graph_repr)
            if self.relu_out:
                if self.out_shape == 1:
                    y_ega_prob = torch.relu(y_ega_prob)
                    y_other_prob = torch.relu(y_other_prob)
                else:
                    y_ega_prob = torch.softmax(y_ega_prob, dim=-1)
                    y_other_prob = torch.softmax(y_other_prob, dim=-1)
            return y_ega_prob, y_other_prob
        else:
            node_repr = self.graph_encoder.get_node_repr(*args, **kwargs)
            graph_repr = self.graph_encoder.encoder.readout(node_repr, kwargs['data'].batch)

            y_ega_prob = self.label_predictor(graph_repr)
            y_ega_pred = y_ega_prob
            return y_ega_pred, y_ega_prob



class DEROGGNN_E(GNNBasic):
    def __init__(self, config, **kwargs):
        super(DEROGGNN_E, self).__init__(config)
        self.final_predictor = PseudoLabelClassifier(config)
        self.env_generator = EnvGenerator(config)
        self.invgraph_encoder = GraphRationaleExtractor(config)
        if config['ood']['env_align']:
            self.env_pred = MLP([config['model']['dim_hidden'], 2 * config['model']['dim_hidden'], config['dataset']['num_envs']],
                                dropout=config['model']['dropout_rate'], config=config, bn=True)
        else:
            self.env_pred = None

        self.config = config
        self.num_layer = config['model']['num_layer']
        self.model_level = config['model']['model_level']
        self.pool_type = config['model']['global_pool']

    def get_prior(self, *args, **kwargs):
        graph_prior = self.env_prior(*args, **kwargs)
        kwargs['graph_repr'] = graph_prior
        node_prior = self.inv_prior(*args, **kwargs)
        return graph_prior, node_prior

    def forward(self, *args, **kwargs):
        y_pred, y_prob = self.final_predictor(*args, **kwargs)
        kwargs['y_pred'], kwargs['y_prob'] = y_pred, y_prob
        graph_repr = self.env_generator(*args, **kwargs)
        kwargs['graph_repr'] = graph_repr
        node_score = self.invgraph_encoder(*args, **kwargs)
        kwargs['node_score'] = node_score

        if self.env_pred:
            env_pred = self.env_pred(graph_repr)
        else:
            env_pred = None
        return y_pred, y_prob, graph_repr, node_score, None, None, env_pred


class DEROGGNN_M(GNNBasic):
    def __init__(self, config, **kwargs):
        super(DEROGGNN_M, self).__init__(config)
        self.rand_env, self.rand_inv = config['ood'].get('rand_env'), config['ood'].get('rand_inv')
        self.egaware_predictor = FinalClassifier(config)

    def forward(self, *args, **kwargs):
        if self.rand_env:
            kwargs['graph_repr'] = torch.rand(kwargs['ood_algorithm'].graph_repr.size(),
                                              device=kwargs['ood_algorithm'].graph_repr.device).detach()
        else:
            kwargs['graph_repr'] = kwargs['ood_algorithm'].graph_repr.detach()
        if self.rand_inv:
            kwargs['node_score'] = torch.rand(kwargs['ood_algorithm'].node_score.size(),
                                              device=kwargs['ood_algorithm'].node_score.device).detach()
        else:
            kwargs['node_score'] = kwargs['ood_algorithm'].node_score.detach()
        y_ega_pred, y_ega_prob = self.egaware_predictor(*args, **kwargs)
        return y_ega_pred, y_ega_prob


class DEROGGNN(nn.Module):
    def __init__(self, config, **kwargs):
        super(DEROGGNN, self).__init__()
        set_seed(config['random_seed'])
        self.e_predictor = DEROGGNN_E(config)
        self.m_predictor = DEROGGNN_M(config)

    def forward(self, *args, **kwargs):
        raise NotImplementedError


class GradientReverseLayerF(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None