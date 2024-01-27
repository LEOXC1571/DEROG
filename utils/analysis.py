import os
from tqdm import tqdm
import time
import numpy as np
import torch
from sklearn.manifold import TSNE
from utils import pbar_setting, nan2zero_get_mask, eval_data_preprocess


def density_analysis(embeddings, seed=42):
    tsne = TSNE(n_components=2, init='random', n_iter=5000, n_iter_without_progress=100,
                method='barnes_hut', random_state=seed).fit_transform(embeddings)
    return tsne

def data_gen(args, path, start_time):
    from utils import load_config, set_seed
    from datasets import load_dataset

    config = load_config(path, start_time, args)
    set_seed(config['random_seed'])
    os.environ['PYTHONHASHSEED'] = str(config['random_seed'])
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # load_logger(config)

    # initialize model and dataset
    print(f"#IN#\n-----------------------------------\n    Task: {config['task']}\n"
          f"{time.asctime(time.localtime(time.time()))}")
    print(f"#IN#Load Dataset {config['dataset']['dataset_name']}")

    dataset = load_dataset(config)
    print(f"#D#Dataset: {dataset}")
    print('#D#', dataset['train'][0] if type(dataset) is dict else dataset[0])

    env_id = dataset['test'].data.domain_id
    # env_id = dataset['test'].data.env_id
    out = {'env_id': env_id}
    torch.save(out, os.path.join(path, f"../oodgen/storage/case/env_{config['dataset']['dataset_name']}_{config['dataset']['domain']}_{config['dataset']['shift_type']}.pt"))


def result_generation(args, path, start_time):
    from utils import load_config, set_seed, eval_data_postprocess
    from datasets import load_dataset, create_loader
    from models import model_map, ood_alg_map

    config = load_config(path, start_time, args)
    set_seed(config['random_seed'])
    os.environ['PYTHONHASHSEED'] = str(config['random_seed'])
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    print(f"#IN#\n-----------------------------------\n    Task: {config['task']}\n"
          f"{time.asctime(time.localtime(time.time()))}")
    print(f"#IN#Load Dataset {config['dataset']['dataset_name']}")

    dataset = load_dataset(config)
    print(f"#D#Dataset: {dataset}")
    print('#D#', dataset['train'][0] if type(dataset) is dict else dataset[0])
    loader = create_loader(dataset, config)
    print('#IN#Loading model...')
    model = model_map[config['model']['model_name']](config).to(config['device'])
    ood_algorithm = ood_alg_map[config['ood']['ood_alg']](config)
    train_time = args.time
    config['test_ckpt'] = os.path.join(config['ckpt_dir'], train_time + '__best.ckpt')
    config['id_test_ckpt'] = os.path.join(config['ckpt_dir'], train_time + '__id_best.ckpt')
    ckpt = torch.load(config['test_ckpt'], map_location=config['device'])
    model.load_state_dict(ckpt['state_dict'])
    # test_split = ['train', 'id_val', 'id_test', 'val', 'test']
    test_split = 'test'
    pred, target, env_repr, inv_repr, total_batch, env_pred = evaluate(config, model, ood_algorithm, loader, test_split, get_env_repr=True, get_inv_repr=True)
    if isinstance(test_split, list):
        out_dict = {}
        for i in range(len(test_split)):
            pred_temp, pred_true_temp = eval_data_postprocess(pred[i], target[i], config['metric'].dataset_task)
            out_dict[test_split[i]] = {
                'pred': torch.tensor(pred_temp),
                'target': torch.tensor(np.concatenate(target[i]), dtype=torch.long),
                'is_true': torch.tensor(pred_true_temp, dtype=torch.bool),
                'env_repr': env_repr[i].cpu(),
                'inv_repr': inv_repr[i].cpu(),
                'node_batch': total_batch[i].cpu()
            }
        torch.save(out_dict, os.path.join(path, f"../oodgen/storage/case/{args.time}{config['ood']['ood_alg']}_{config['dataset']['dataset_name']}_{config['dataset']['domain']}_{config['dataset']['shift_type']}_{len(test_split)}split.pt"))
    else:
        pred, pred_true = eval_data_postprocess(pred, target, config['metric'].dataset_task)

        out_dict = {
            'pred': torch.tensor(pred),
            'target': torch.tensor(np.concatenate(target), dtype=torch.long),
            'is_true': torch.tensor(pred_true, dtype=torch.bool),
            'env_repr': env_repr.cpu(),
            'inv_repr': inv_repr.cpu(),
            'node_batch': total_batch.cpu()
        }
        torch.save(out_dict, os.path.join(path, f"../oodgen/storage/case/{args.time}{config['ood']['ood_alg']}_{config['dataset']['dataset_name']}_{config['dataset']['domain']}_{config['dataset']['shift_type']}.pt"))


@torch.no_grad()
def evaluate(config, model, ood_algorithm, loader, split, get_env_repr=False, get_inv_repr=False):
    if isinstance(split,list):
        pred_total, target_total, env_repr_total, inv_repr_total, batch_total = [], [], [], [], []
        for split_type in split:
            model.eval()
            pred_all = []
            target_all = []
            env_repr = [] if get_env_repr else None
            inv_repr = [] if get_inv_repr else None
            total_batch = [] if get_inv_repr else None
            pbar = tqdm(loader[split_type], desc=f'Eval {split_type.capitalize()}', total=len(loader[split_type]), **pbar_setting)
            for data in pbar:
                data = data.to(config['device'])
                mask, targets = nan2zero_get_mask(data, split_type, config)
                node_norm = (torch.ones((data.num_nodes,), device=config['device'])
                             if config['model']['model_level'] == 'node' else None)
                data, targets, mask, node_norm = ood_algorithm.input_preprocess(data, targets, mask, node_norm, model.training, config)
                e_output = model.e_predictor(data=data, edge_weight=None, ood_algorithm=ood_algorithm)
                raw_e_pred = ood_algorithm.output_postprocess_e(e_output)
                m_output = model.m_predictor(data=data, edge_weight=None, ood_algorithm=ood_algorithm)
                raw_preds = ood_algorithm.output_postprocess_m(m_output)
                pred, target = eval_data_preprocess(data.y, raw_preds, mask, config['metric'].dataset_task)
                pred_all.append(pred)
                target_all.append(target)
                env_repr.append(ood_algorithm.graph_repr) if get_env_repr else None
                total_batch.append(data.batch + config['train']['test_bs'] * len(inv_repr)) if get_inv_repr else None
                inv_repr.append(ood_algorithm.node_score) if get_inv_repr else None
            env_repr = torch.cat(env_repr, dim=0) if get_env_repr else None
            inv_repr = torch.cat(inv_repr, dim=0) if get_inv_repr else None
            total_batch = torch.cat(total_batch, dim=0) if get_inv_repr else None

            pred_total.append(pred_all)
            target_total.append(target_all)
            env_repr_total.append(env_repr)
            inv_repr_total.append(inv_repr)
            batch_total.append(total_batch)
        return pred_total, target_total, env_repr_total, inv_repr_total, batch_total
    else:
        model.eval()
        pred_all = []
        target_all = []
        env_repr = [] if get_env_repr else None
        inv_repr = [] if get_inv_repr else None
        total_batch = [] if get_inv_repr else None
        env_pred = []
        pbar = tqdm(loader[split], desc=f'Eval {split.capitalize()}', total=len(loader[split]), **pbar_setting)
        for data in pbar:
            data = data.to(config['device'])
            mask, targets = nan2zero_get_mask(data, split, config)
            node_norm = (torch.ones((data.num_nodes,), device=config['device'])
                         if config['model']['model_level'] == 'node' else None)
            data, targets, mask, node_norm = ood_algorithm.input_preprocess(data, targets, mask, node_norm,
                                                                            model.training, config)
            e_output = model.e_predictor(data=data, edge_weight=None, ood_algorithm=ood_algorithm)
            raw_e_pred = ood_algorithm.output_postprocess_e(e_output)
            m_output = model.m_predictor(data=data, edge_weight=None, ood_algorithm=ood_algorithm)
            raw_preds = ood_algorithm.output_postprocess_m(m_output)
            pred, target = eval_data_preprocess(data.y, raw_preds, mask, config['metric'].dataset_task)
            env_pred.append(ood_algorithm.env_pred)
            pred_all.append(pred)
            target_all.append(target)
            env_repr.append(ood_algorithm.graph_repr) if get_env_repr else None
            total_batch.append(data.batch + config['train']['test_bs'] * len(inv_repr)) if get_inv_repr else None
            inv_repr.append(ood_algorithm.node_score) if get_inv_repr else None
        env_repr = torch.cat(env_repr, dim=0) if get_env_repr else None
        inv_repr = torch.cat(inv_repr, dim=0) if get_inv_repr else None
        total_batch = torch.cat(total_batch, dim=0) if get_inv_repr else None
        return pred_all, target_all, env_repr, inv_repr, total_batch, env_pred



