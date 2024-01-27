from torch_geometric.loader import DataLoader, GraphSAINTRandomWalkSampler

from .good_cmnist import GOODCMNIST
from .good_hiv import GOODHIV
from .good_motif import GOODMotif
from .good_sst2 import GOODSST2
from .good_twitter import GOODTwitter
from .lbap_core import LBAPcore
from utils import set_seed, worker_init_fn


dataset_map = {
    'GOODCMNIST': GOODCMNIST,
    'GOODHIV': GOODHIV,
    'GOODMotif': GOODMotif,
    'GOODSST2': GOODSST2,
    'GOODTWITTER': GOODTwitter,
    'LBAPcore': LBAPcore
}


class BaseDataLoader(dict):
    def __init__(self, *args, **kwargs):
        super(BaseDataLoader, self).__init__(*args, **kwargs)

    @classmethod
    def setup(cls, dataset, config):
        set_seed(config['random_seed'])
        if config['model']['model_level'] == 'node':
            graph = dataset[0]
            loader = GraphSAINTRandomWalkSampler(graph, batch_size=config['train']['train_bs'],
                                                 walk_length=config['model']['model_layer'],
                                                 num_steps=config['train']['num_steps'], sample_coverage=100,
                                                 save_dir=dataset.processed_dir)
            if config['ood']['ood_alg'] == 'EERM':
                loader = {'train': [graph], 'eval_train': [graph], 'id_val': [graph], 'id_test': [graph], 'val': [graph],
                          'test': [graph]}
            else:
                loader = {'train': loader, 'eval_train': [graph], 'id_val': [graph], 'id_test': [graph], 'val': [graph],
                          'test': [graph]}
        else:
            loader = {'train': DataLoader(dataset['train'], batch_size=config['train']['train_bs'], shuffle=True, num_workers=config['num_workers'], pin_memory=True),
                      'eval_train': DataLoader(dataset['train'], batch_size=config['train']['val_bs'], shuffle=False, num_workers=config['num_workers'], pin_memory=True),
                      'id_val': DataLoader(dataset['id_val'], batch_size=config['train']['val_bs'], shuffle=False, num_workers=config['num_workers'], pin_memory=True) if dataset.get(
                          'id_val') else None,
                      'id_test': DataLoader(dataset['id_test'], batch_size=config['train']['test_bs'],
                                            shuffle=False, num_workers=config['num_workers'], pin_memory=True) if dataset.get(
                          'id_test') else None,
                      'val': DataLoader(dataset['val'], batch_size=config['train']['val_bs'], shuffle=False, num_workers=config['num_workers'], pin_memory=True),
                      'test': DataLoader(dataset['test'], batch_size=config['train']['test_bs'], shuffle=False, num_workers=config['num_workers'], pin_memory=True)}

        return cls(loader)


def load_dataset(config):
    try:
        set_seed(config['random_seed'])
        dataset, meta_info = dataset_map[config['dataset']['dataset_name']].load(
            dataset_root=config['dataset']['dataset_root'],
            domain=config['dataset']['domain'],
            shift=config['dataset']['shift_type'],
            generate=config['dataset']['generate']
        )
    except KeyError as e:
        print('Dataset not found!')
        raise e

    config['dataset']['dataset_type'] = meta_info['dataset_type']
    config['model']['model_level'] = meta_info['model_level']
    config['dataset']['dim_node'] = meta_info['dim_node']
    config['dataset']['dim_edge'] = meta_info['dim_edge']
    config['dataset']['num_envs'] = meta_info['num_envs']
    config['dataset']['num_classes'] = meta_info['num_classes']
    config['dataset']['num_train_nodes'] = meta_info.get('num_train_nodes')
    config['dataset']['num_domains'] = meta_info.get('num_domains')
    config['dataset']['feat_dims'] = meta_info.get('feat_dims')
    config['dataset']['edge_feat_dims'] = meta_info.get('edge_feat_dims')

    config['metric'].set_score_func(dataset['metric'] if type(dataset) is dict else getattr(dataset, 'metric'))
    config['metric'].set_loss_func(dataset['task']) if type(dataset) is dict else getattr(dataset, 'task')

    return dataset


def create_loader(dataset, config):
    loader = BaseDataLoader.setup(dataset, config)
    return loader