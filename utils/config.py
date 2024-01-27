import os
import copy
import torch
from pathlib import Path
from ruamel.yaml import YAML
from os.path import join as opj
from .metric import Metric


domain_sum = {
    'cmnist': {
        'name': 'GOODCMNIST',
        'domain': ['color', 'background'],
        'shift': ['covariate', 'concept']
    },
    'motif': {
        'name': 'GOODMotif',
        'domain': ['basis', 'size'],
        'shift': ['covariate', 'concept']
    },
    'hiv': {
        'name': 'GOODHIV',
        'domain': ['scaffold', 'size'],
        'shift': ['covariate', 'concept']
    },
    'twitter': {
        'name': 'GOODTWITTER',
        'domain': ['length'],
        'shift': ['covariate', 'concept']
    },
    'sst2': {
        'name': 'GOODSST2',
        'domain': ['length'],
        'shift': ['covariate', 'concept']
    },
    'lbap': {
        'name': 'LBAPcore',
        'domain': ['scaffold', 'size', 'assay'],
        'shift': ['covariate']
    }
}


def merge_dicts(dict1: dict, dict2: dict):
    if not isinstance(dict1, dict):
        raise ValueError(f"Expecting dict1 to be dict, found {type(dict1)}.")
    if not isinstance(dict2, dict):
        raise ValueError(f"Expecting dict2 to be dict, found {type(dict2)}.")

    return_dict = copy.deepcopy(dict1)
    duplicates = []

    for k, v in dict2.items():
        if k not in dict1:
            return_dict[k] = v
        else:
            if isinstance(v, dict) and isinstance(dict1[k], dict):
                return_dict[k], duplicates_k = merge_dicts(dict1[k], dict2[k])
                duplicates += [f"{k}.{dup}" for dup in duplicates_k]
            else:
                return_dict[k] = dict2[k]
                duplicates.append(k)

    return return_dict, duplicates


def load_config(path, time, args, previous_includes=[]):
    data = args.data
    model = args.model
    domain = args.domain
    shift = args.shift
    gpu = args.gpu

    root_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    def read_config(path, data=None, model=None, domain=None, shift='covariate', previous_includes=[]):
        if data is not None and model is not None:
            if domain not in domain_sum[data]['domain']:
                domain = domain_sum[data]['domain'][0]
            path = os.path.join(path, 'configs', domain_sum[data.lower()]['name'], domain, shift, model.upper() + '.yaml')
        path = Path(path)
        previous_includes = previous_includes + [path]
        yaml = YAML(typ='safe')
        direct_config = yaml.load(open(path, 'r'))
        if "includes" in direct_config:
            includes = direct_config.pop("includes")
        else:
            includes = []
        if not isinstance(includes, list):
            raise AttributeError(
                "Includes must be a list, '{}' provided".format(type(includes))
            )

        config = {}
        duplicates_warning = []
        duplicates_error = []

        for include in includes:
            include = path.parent / include
            include_config, inc_dup_warning, inc_dup_error = read_config(
                include, previous_includes=previous_includes
            )
            duplicates_warning += inc_dup_warning
            duplicates_error += inc_dup_error

            # Duplicates between includes causes an error
            config, merge_dup_error = merge_dicts(config, include_config)
            duplicates_error += merge_dup_error

        # Duplicates between included and main file causes warnings
        config, merge_dup_warning = merge_dicts(config, direct_config)
        duplicates_warning += merge_dup_warning
        return config, duplicates_warning, duplicates_error

    config, _, _ = read_config(path, data, model, domain, shift, previous_includes)
    STORAGE_DIR = opj(os.path.abspath(root_path), 'storage')
    if config['dataset']['dataset_root'] is None:
        config['dataset']['dataset_root'] = opj(STORAGE_DIR, 'datasets')

    # --- tensorboard directory setting ---
    config['tensorboard_logdir'] = opj(STORAGE_DIR, 'tensorboard', f"{config['dataset']['dataset_name']}")
    if config['dataset']['shift_type']:
        config['tensorboard_logdir'] = opj(config['tensorboard_logdir'], config['dataset']['shift_type'],
                                           config['ood']['ood_alg'], str(config['ood']['ood_param']))

    # --- Round setting ---
    # if config['exp_round']:
    config['random_seed'] = config['random_seed']

    # --- Directory name definitions ---
    dataset_dirname = config['dataset']['dataset_name'] + '_' + config['dataset']['domain']
    if config['dataset']['shift_type']:
        dataset_dirname += '_' + config['dataset']['shift_type']
    model_dirname = f"{config['model']['model_name']}"
    # model_dirname = f"{config['model']['model_name']}_{config['model']['model_layer']}l_{config['model']['global_pool']}pool_{config['model']['dropout_rate']}dp"
    train_dirname = f"{config['train']['lr']}lr_{config['train']['weight_decay']}wd"
    ood_dirname = config['ood']['ood_alg']
    if config['ood']['ood_param'] is not None and config['ood']['ood_param'] >= 0:
        ood_dirname += f"_{config['ood']['ood_param']}"
    else:
        ood_dirname += '_no_param'
    if config['ood']['extra_param'] is not None:
        for i, param in enumerate(config['ood']['extra_param']):
            ood_dirname += f'_{param}'

    # --- Log setting ---
    # log_dir_root = opj(root_path, 'log', 'round' + str(config['exp_round']))
    log_dir_root = opj(root_path, 'log')
    log_dirs = opj(log_dir_root, dataset_dirname, model_dirname)
    if config['save_tag']:
        log_dirs = opj(log_dirs, config['save_tag'])
    config['log_path'] = opj(log_dirs, f'{time}.log')
    # config['log_path'] = opj(log_dirs, f'{time}_{train_dirname}_{ood_dirname}_{config["log_file"]}.log')

    # --- Checkpoint setting ---
    if config['ckpt_root'] is None:
        config['ckpt_root'] = opj(root_path, 'checkpoints')
    if config['ckpt_dir'] is None:
        config['ckpt_dir'] = opj(config['ckpt_root'], 'round' + str(config['exp_round']))
        config['ckpt_dir'] = opj(config['ckpt_dir'], dataset_dirname, model_dirname)
        # config['ckpt_dir'] = opj(config['ckpt_dir'], dataset_dirname, model_dirname, train_dirname, ood_dirname)
        if config['save_tag']:
            config['ckpt_dir'] = opj(config['ckpt_dir'], config['save_tag'])
    config['test_ckpt'] = opj(config['ckpt_dir'], f'{time}_best.ckpt')
    config['id_test_ckpt'] = opj(config['ckpt_dir'], f'{time}_id_best.ckpt')

    # --- Other settings ---
    if config['train']['max_epoch'] > 1000:
        config['train']['save_gap'] = config['train']['max_epoch'] // 100
    config['gpu_idx'] = gpu
    config['device'] = torch.device(f"cuda:{config['gpu_idx']}" if torch.cuda.is_available() else 'cpu')
    config['train']['stage_stones'].append(100000)
    config['metric'] = Metric()
    config['comment'] = args.comment
    config['time'] = time

    return config

