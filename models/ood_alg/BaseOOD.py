from abc import ABC
import torch
import torch.nn as nn
import torch.optim as optim

from utils import set_seed, at_stage


class BaseOODAlg(ABC):
    def __init__(self, config):
        super(BaseOODAlg, self).__init__()
        self.optimizer: optim.Adam = None
        self.scheduler: optim.lr_scheduler._LRScheduler = None
        self.model: nn.Module = None

        self.mean_loss = None
        self.spec_loss = None
        self.stage = 0

    def stage_control(self, config):
        if self.stage == 0 and at_stage(1, config):
            set_seed(config['random_seed'])
            self.stage = 1

    def input_preprocess(self, data, targets, mask, node_norm, training, config, **kwargs):
        return data, targets, mask, node_norm

    def output_postprocess(self, model_output, **kwargs):
        return model_output

    def loss_calculate(self, raw_pred, targets, mask, node_norm, config):
        loss = config['metric'].loss_func(raw_pred, targets, reduction='none') * mask
        loss = loss * node_norm * mask.sum() if config['model']['model_level'] == 'node' else loss
        return loss

    def loss_postprocess(self, loss, data, mask, config, **kwargs):
        self.mean_loss = loss.sum() / mask.sum()
        return self.mean_loss

    def set_up(self, model, config):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['train']['lr'],
                                          weight_decay=config['train']['weight_decay'])
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=config['train']['mile_stones'],
                                                        gamma=0.1)

    def backward(self, loss):
        loss.backward()
        self.optimizer.step()
