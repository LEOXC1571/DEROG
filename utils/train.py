import random
import os
import numpy as np
import torch
from torch.distributions.normal import Normal

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.enable_grad()

def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def nan2zero_get_mask(data, task, config):
    if config['model']['model_level'] == 'node':
        if 'train' in task:
            mask = data.train_mask
        elif task == 'id_val':
            mask = data.get('id_val_mask')
        elif task == 'id_test':
            mask = data.get('id_test_mask')
        elif task == 'val':
            mask = data.val_mask
        elif task == 'test':
            mask = data.test_mask
        else:
            raise ValueError(f'Task should be train/id_val/id_test/val/test, but got {task}.')
    else:
        mask = ~torch.isnan(data.y)
    if mask is None:
        return None, None
    targets = torch.clone(data.y).detach()
    targets[~mask] = 0

    return mask, targets


def at_stage(i, config):
    if i - 1 < 0:
        raise ValueError(f"Stage i must be equal or larger than 0, but got {i}.")
    if i > len(config['train']['stage_stones']):
        raise ValueError(f"Stage i should be smaller than the largest stage {len(config['train']['stage_stones'])},"
                         f"but got {i}.")
    if i - 2 < 0:
        return config['train']['epoch'] <= config['train']['stage_stones'][i - 1]
    else:
        return config['train']['stage_stones'][i - 2] < config['train']['epoch'] <= config['train']['stage_stones'][i - 1]

def discrete_gaussian(nums, std=1):
    Dist = Normal(loc=0, scale=1)
    plen, halflen = std * 6 / nums, std * 3 / nums
    posx = torch.arange(-3 * std + halflen, 3 * std, plen)
    result = Dist.cdf(posx + halflen) - Dist.cdf(posx - halflen)
    return result / result.sum()


def get_uniform(num):
    prior = torch.ones(num) / num
    return prior

def torch_gaussian(num, device):
    return Normal(loc=0, scale=1).sample((num,)).to(device).abs()

def get_prior(config):
    if config['ood']['env_prior'] == 'dis_gau':
        env_prior = discrete_gaussian(config['model']['dim_hidden']).to(config['device'])
    elif config['ood']['env_prior'] == 'con_gau':
        env_prior = torch_gaussian(config['model']['dim_hidden'], config['device'])
    elif config['ood']['env_prior'] == 'uniform':
        env_prior = get_uniform(config['model']['dim_hidden']).to(config['device'])
    else:
        env_prior = None

    if config['ood']['inv_prior'] == 'dis_gau':
        inv_prior = discrete_gaussian(config['model']['dim_hidden']).to(config['device'])
    elif config['ood']['inv_prior'] == 'con_gau':
        inv_prior = torch_gaussian(config['model']['dim_hidden'], config['device'])
    elif config['ood']['inv_prior'] == 'uniform':
        inv_prior = get_uniform(config['model']['dim_hidden']).to(config['device'])
    else:
        inv_prior = None
    return env_prior, inv_prior