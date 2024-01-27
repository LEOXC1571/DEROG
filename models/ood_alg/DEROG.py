from typing import Tuple
import numpy as np
import random
import torch
from torch_geometric.utils import to_dense_batch
import torch.optim as optim
from .BaseOOD import BaseOODAlg
from collections import OrderedDict
from utils import at_stage, set_seed, get_prior
import torch.nn.functional as F


class DEROG(BaseOODAlg):
    def __init__(self, config):
        super(DEROG, self).__init__(config)
        self.att = None
        self.edge_att = None
        self.targets = None

        self.spec_loss = OrderedDict()
        self.optimizer_e = None
        self.optimizer_m = None

        self.env_prior, self.inv_prior = get_prior(config)
        self.kl = config['metric'].kl_div

        self.decay_r = 0.1
        self.decay_interval = config['ood']['decay_interval']

    def set_up(self, model, config):
        self.model = model
        self.optimizer_e = optim.Adam(self.model.e_predictor.parameters(), lr=config['train']['lr'],
                                      weight_decay=config['train']['weight_decay'])
        self.scheduler_e = optim.lr_scheduler.MultiStepLR(self.optimizer_e, milestones=config['train']['mile_stones'],
                                                          gamma=0.1)
        self.optimizer_m = optim.Adam(self.model.m_predictor.parameters(), lr=config['train']['lr'],
                                      weight_decay=config['train']['weight_decay'])
        self.scheduler_m = optim.lr_scheduler.MultiStepLR(self.optimizer_m, milestones=config['train']['mile_stones'],
                                                          gamma=0.1)

    def stage_control(self, config):
        if self.stage == 0 and at_stage(1, config):
            set_seed(config['random_seed'])
            self.stage = 1

    def backward_e(self, loss):
        loss.backward()
        self.optimizer_e.step()
        self.optimizer_e.zero_grad()

    def backward_m(self, loss):
        loss.backward()
        self.optimizer_m.step()
        self.optimizer_m.zero_grad()

    def output_postprocess(self, model_output, **kwargs):
        self.y_pred, self.y_prob, self.graph_repr, self.node_score, self.y_ega_pred, self.y_ega_prob = model_output
        return self.y_pred

    def output_postprocess_e(self, model_output, **kwargs):
        self.y_pred, self.y_prob, self.graph_repr, self.node_score, _, _, self.env_pred = model_output
        return self.y_pred

    def output_postprocess_e_temp(self, model_output, **kwargs):
        _, _, self.graph_repr, self.node_score, env_prior, inv_prior, _ = model_output
        if self.env_prior is None:
            self.env_prior = env_prior
        if self.inv_prior is None:
            self.inv_prior = inv_prior

    def output_postprocess_m(self, model_output, **kwargs):
        self.y_ega_pred, self.y_ega_prob = model_output
        return self.y_ega_pred

    def loss_calculate(self, raw_pred, targets, mask, node_norm, config):
        loss = config['metric'].loss_func(raw_pred, targets, reduction='none') * mask
        self.targets = targets
        return loss

    def loss_calculate_e(self, raw_pred, targets, mask, node_norm, config):

        if config['ood']['e_out'] == 'entropy':
            loss = config['ood']['weight_e_out'] * (raw_pred+1e-6) * torch.log(raw_pred+1e-6) * mask
        else:
            loss = config['ood']['weight_e_out'] * config['metric'].loss_func(raw_pred, targets, reduction='none') * mask
        self.targets = targets
        return loss

    def loss_calculate_m(self, raw_pred, targets, mask, node_norm, config):
        loss = config['metric'].loss_func(raw_pred, targets, reduction='none') * mask
        self.targets = targets
        return loss
    def loss_postprocess_e(self, loss, data, mask, config, **kwargs):
        if self.env_pred is not None:
            self.spec_loss['env_align'] = config['ood']['env_align_weight'] * config['metric'].cross_entropy_with_logit(self.env_pred, data.env_id, reduction='mean')
        if config['model']['augment_inv']:
            self.spec_loss['con_inv'] = config['ood']['weight_cl'] * self.contrastive_inv_loss(self.node_score, data, config)
        if config['ood']['weight_kl_env']:
            self.spec_loss['env_kl'] = config['ood']['weight_kl_env'] * config['metric'].kl(self.graph_repr, self.env_prior, type=config['ood']['kl_type']) / mask.sum()
            self.spec_loss['inv_kl'] = config['ood']['weight_kl_inv'] * config['metric'].kl(self.node_score, self.inv_prior, type=config['ood']['kl_type']) / mask.sum()

        self.mean_loss_e = loss.sum() / mask.sum()
        loss = self.mean_loss_e + sum(self.spec_loss.values())
        return loss

    def loss_postprocess_m(self, loss, data, mask, config, **kwargs):
        self.mean_loss_m = loss.sum() / mask.sum()
        loss = self.mean_loss_m
        self.mean_loss = (self.mean_loss_e + self.mean_loss_m).item()
        return loss

    def contrastive_inv_loss(self, node_score, data, config):
        if config['ood'].get('neg_sample'):
            neg_sample = config['ood'].get('neg_sample')
        else:
            neg_sample = 2
        dense_score, mask = to_dense_batch(node_score, data.batch)
        single_node_score = dense_score.sum(dim=-1)
        sort_score, indice = single_node_score.sort(descending=True, dim=1)
        num_nodes = mask.sum(-1)
        pos_rand, neg_rand = random.uniform(0, 0.5), random.uniform(0.5, 1)
        pos_spl1, pos_spl2 = (pos_rand * num_nodes).ceil().long(), (pos_rand * num_nodes).floor().long()
        if neg_sample == 2:
            neg_spl1, neg_spl2 = (neg_rand * num_nodes).ceil().long(), (neg_rand * num_nodes).floor().long()
            if (neg_spl1 > num_nodes-1).sum() > 0:
                neg_spl1, neg_spl2 = neg_spl1 - 1, neg_spl2 - 1


            batch_idx = torch.tensor(range(mask.size(0)))
            pos_idx1, pos_idx2 = indice[batch_idx, pos_spl1], indice[batch_idx, pos_spl2]
            neg_idx1, neg_idx2 = indice[batch_idx, neg_spl1], indice[batch_idx, neg_spl2]

            pos_score1, pos_score2 = dense_score[batch_idx, pos_idx1], dense_score[batch_idx, pos_idx2]
            neg_score1, neg_score2 = dense_score[batch_idx, neg_idx1], dense_score[batch_idx, neg_idx2]

            pos_score1, pos_score2 = F.normalize(pos_score1, dim=1), F.normalize(pos_score2, dim=1)
            neg_score1, neg_score2 = F.normalize(neg_score1, dim=1), F.normalize(neg_score2, dim=1)
            v1 = (pos_score1 * pos_score2).sum(1)
            v1 = torch.exp(v1 / config['model']['augment_tau'])
            v2 = (pos_score1 * neg_score1).sum(1)
            v2 = v2 + (pos_score1 * neg_score2).sum(1)
            v2 = torch.exp(v2 / config['model']['augment_tau'])
            loss = -torch.sum(torch.log(v1 / v2)) / mask.sum()
        else:
            mid = (0.5 * num_nodes).floor().long()
            neg_score_ls = []
            for idx in range(len(num_nodes)):
                if mid[idx] >= neg_sample:
                    index = torch.LongTensor(random.sample(range(mid[idx], num_nodes[idx]), neg_sample)).to(dense_score.device)
                else:
                    index = torch.LongTensor(np.random.choice(range(mid[idx], num_nodes[idx]), neg_sample)).to(dense_score.device)
                neg_score = torch.index_select(dense_score[idx, :, :].unsqueeze(0), 1, index)
                neg_score_ls.append(neg_score)
            neg_score = torch.cat(neg_score_ls)

            batch_idx = torch.tensor(range(mask.size(0)))
            pos_idx1, pos_idx2 = indice[batch_idx, pos_spl1], indice[batch_idx, pos_spl2]
            pos_score1, pos_score2 = dense_score[batch_idx, pos_idx1], dense_score[batch_idx, pos_idx2]
            pos_score1, pos_score2 = F.normalize(pos_score1, dim=1), F.normalize(pos_score2, dim=1)
            neg_score = F.normalize(neg_score, dim=2)

            v1 = (pos_score1 * pos_score2).sum(1)
            v1 = torch.exp(v1 / config['model']['augment_tau'])
            v2 = (pos_score1.unsqueeze(1) * neg_score).sum(1).sum(1)
            v2 = torch.exp(v2 / config['model']['augment_tau'])
            loss = -torch.sum(torch.log(v1 / v2)) / mask.sum()

        return loss


