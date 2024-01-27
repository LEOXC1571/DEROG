import os
from datetime import datetime

from cilog import create_logger
from torch.utils.tensorboard import SummaryWriter


color_palette = ['#FF4359', '#FF8A25', '#FFD75F', '#0078FF', '#00ECC2']
pbar_setting = {'colour': '#a48fff', 'bar_format': '{l_bar}{bar:20}{r_bar}',
                'dynamic_ncols': True, 'ascii': '░▒█'}

important_keys = ['comment', 'random_seed', 'train', 'model', 'dataset', 'ood']

def load_logger(config, sub_print=True):
    if sub_print:
        print("This logger will substitute general print function")
    logger = create_logger(name='GNN_log',
                           file=config['log_path'],
                           enable_mail=False,
                           sub_print=sub_print)

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    for key in important_keys:
        print(f'#IN# {key}: {config[key]}')
    writer = SummaryWriter(
        log_dir=os.path.join(config['tensorboard_logdir'], f"{config['log_file']}_{current_time}"))
    return logger, writer
