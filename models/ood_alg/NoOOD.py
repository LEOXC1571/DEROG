from typing import Tuple
import torch
from torch_geometric.data import Batch

from .BaseOOD import BaseOODAlg
from collections import OrderedDict


class NoOOD(BaseOODAlg):
    def __init__(self, config):
        super(NoOOD, self).__init__(config)
        self.att = None
        self.edge_att = None
        self.targets = None

        self.decay_r = 0.1
        self.decay_interval = config['ood']['decay_interval']

    def output_postprocess(self, model_output, **kwargs):
        raw_output = model_output
        return raw_output

    def loss_calculate(self, raw_pred, targets, mask, node_norm, config):
        loss = config['metric'].loss_func(raw_pred, targets, reduction='none') * mask
        self.targets = targets
        return loss

    def loss_postprocess(self, loss, data, mask, config, **kwargs):
        self.mean_loss = loss.sum() / mask.sum()
        return self.mean_loss
