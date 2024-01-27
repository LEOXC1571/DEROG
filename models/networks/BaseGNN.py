import torch
import torch.nn as nn
from torch_geometric.data.batch import Batch
from .Pooling import GlobalMeanPool, GlobalMaxPool, IdenticalPool
from torch.nn import Identity
from utils import set_seed


class GNNBasic(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super(GNNBasic, self).__init__()
        self.config = config
        set_seed(config['random_seed'])

    def arguments_read(self, *args, **kwargs):
        data: Batch = kwargs.get('data') or None

        if not data:
            if not args:
                assert 'x' in kwargs
                assert 'edge_index' in kwargs
                x, edge_index = kwargs['x'], kwargs['edge_index'],
                batch = kwargs.get('batch')
                if batch is None:
                    batch = torch.zeros(kwargs['x'].shape[0], dtype=torch.int64, device=torch.device('cuda'))
            elif len(args) == 2:
                x, edge_index, batch = args[0], args[1], \
                                       torch.zeros(args[0].shape[0], dtype=torch.int64, device=torch.device('cuda'))
            elif len(args) == 3:
                x, edge_index, batch = args[0], args[1], args[2]
            else:
                raise ValueError(f"forward's args should take 2 or 3 arguments but got {len(args)}")
        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch

        if self.config['model']['model_level'] != 'node':
            # --- Maybe batch size --- Reason: some method may filter graphs leading inconsistent of batch size
            batch_size: int = kwargs.get('batch_size') or (batch[-1].item() + 1)

        if self.config['model']['model_level'] == 'node':
            edge_weight = kwargs.get('edge_weight')
            return x, edge_index, edge_weight, batch
        elif self.config['dataset']['dim_edge'] or kwargs.get('edge_feat'):
            edge_attr = data.edge_attr
            return x, edge_index, edge_attr, batch, batch_size

        return x, edge_index, batch, batch_size

    def probs(self, *args, **kwargs):
        # nodes x classes
        return self(*args, **kwargs).softmax(dim=1)

    def at_stage(self, i):
        if i - 1 < 0:
            raise ValueError(f"Stage i must be equal or larger than 0, but got {i}.")
        if i > len(self.config['train']['stage_stones']):
            raise ValueError(f"Stage i should be smaller than the largest stage"
                             f" {len(self.config['train']['stage_stones'])},"
                             f"but got {i}.")
        if i - 2 < 0:
            return self.config['train']['epoch'] < self.config['train']['stage_stones'][i - 1]
        else:
            return (self.config['train']['stage_stones'][i - 2] <= self.config['train']['epoch'] <
                    self.config['train']['stage_stones'][i - 1])


class BasicEncoder(nn.Module):
    def __init__(self, config, **kwargs):
        if type(self).mro()[type(self).mro().index(__class__) + 1] is nn.Module:
            super(BasicEncoder, self).__init__()
        else:
            super(BasicEncoder, self).__init__(config)
        num_layer = config['model']['num_layer']

        self.relus = nn.ModuleList(
            [
                nn.ReLU()
                for _ in range(num_layer)
            ]
        )
        if kwargs.get('no_bn'):
            self.batch_norms = [
                Identity()
                for _ in range(num_layer)
            ]
        else:
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(config['model']['dim_hidden'], track_running_stats=True)
                for _ in range(num_layer)
            ])
        self.dropouts = nn.ModuleList([
            nn.Dropout(config['model']['dropout_rate'])
            for _ in range(num_layer)
        ])
        if config['model']['model_level'] == 'node':
            self.readout = IdenticalPool()
        elif config['model']['global_pool'] == 'mean':
            self.readout = GlobalMeanPool()
        elif config['model']['global_pool'] == 'max':
            self.readout = GlobalMaxPool()
        elif config['model']['global_pool'] == 'id':
            self.readout = IdenticalPool()
        else:
            self.readout = GlobalMaxPool()
