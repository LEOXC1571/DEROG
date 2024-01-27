import os
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
import argparse
import time
start_time = time.strftime('%Y-%m-%d_%H-%M-%S_', time.localtime())
from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE,SIG_DFL)


from datasets import load_dataset, create_loader
from kernels import pipeline_map
from models import model_map, ood_alg_map
from utils import load_config, load_logger, set_seed



def main(args):
    config = load_config(CURRENT_PATH, start_time, args)
    set_seed(config['random_seed'])
    os.environ['PYTHONHASHSEED'] = str(config['random_seed'])
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # torch.use_deterministic_algorithms(True)
    load_logger(config)

    # initialize model and dataset
    print(f"#IN#\n-----------------------------------\n    Task: {config['task']}\n"
          f"{time.asctime(time.localtime(time.time()))}")
    print(f"#IN#Load Dataset {config['dataset']['dataset_name']}")
    dataset = load_dataset(config)
    print(f"#D#Dataset: {dataset}")
    print('#D#', dataset['train'][0] if type(dataset) is dict else dataset[0])
    loader = create_loader(dataset, config)
    print('#IN#Loading model...')
    model = model_map[config['model']['model_name']](config)
    ood_algorithm = ood_alg_map[config['ood']['ood_alg']](config)
    pipeline = pipeline_map[config['pipeline']](config['task'], model, loader, ood_algorithm, config)
    pipeline.load_task()
    if config['task'] == 'train':
        pipeline.task = 'test'
        pipeline.load_task()
    print('Finished')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='derog',
                        help='model_name')
    parser.add_argument('--data', type=str, default='hiv',
                        help='dataset_name')
    parser.add_argument('--gpu', type=int, default='0',
                        help='gpu_id')
    parser.add_argument('--domain', type=str, default='scaffold',
                        help='domain_type')
    parser.add_argument('--shift', type=str, default='covariate',
                        help='covariate_type')
    parser.add_argument('--comment', type=str, default='None', action='store',
                        help='comment on the experiment')

    args, unknown = parser.parse_known_args()
    main(args)

