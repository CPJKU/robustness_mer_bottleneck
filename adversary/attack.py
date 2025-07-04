import yaml
import torch
import random
import argparse
import numpy as np
import torch.nn as nn

from glob import glob
from pathlib import Path
from torch.utils.data import DataLoader

from paths import RUN_DIR, HOME_DIR
from adversary.attacker import Attacker
from data.emotion import EmotionDataset
from train_models.utils import load_model
from train_models.train_funcs import test
from train_models.train import get_configured_model_architecture


def opts_parser():
    """ Command line argument setup. """
    parser = argparse.ArgumentParser(description='Script that attacks music emotion recognition system.')
    parser.add_argument('--exp_dir', type=str, required=True,
                        help='Directory containing trained model that should be attacked.')
    parser.add_argument('--name', required=True, type=str,
                        help='Name of this adversarial experiment, used to name directory in which experiment is logged.')
    parser.add_argument('--num_workers', type=int, required=False, default=8,
                        help='Number of workers used for data loading.')
    parser.add_argument('--seed', type=int, default=21, help='Random seed.')
    parser.add_argument('--attack_test', action='store_true',
                        help='If listed, attack is performed on test set, on validation data otherwise.')
    return parser


def set_random_seed(seed: int):
    """ Sets random seeds for `torch`, `numpy` and `random`. """
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    np.random.seed(seed)
    random.seed(seed)


def get_optim_class(optimiser: str):
    """ Given name of optimiser, returns according `torch.optim.Optimizer` object. """
    if hasattr(torch.optim, optimiser):
        optim = getattr(torch.optim, optimiser)
    else:
        raise ValueError('Please define valid optimiser, {} does not exist'.format(optimiser))
    return optim


def get_model_config(config_name: str):
    """ Parses model configuration string, returns dict containing config. """
    dict = {}
    split = config_name.split('=')

    for i in range(0, len(split) - 1):
        key_parts = split[i].split('_')
        if len(key_parts) > 1 and 'seed' not in split[i]:
            key = '_'.join(key_parts[1:])
        elif 'seed' in split[i]:
            key = key_parts[-1]
        else:
            key = key_parts[0]

        value_parts = split[i + 1].split('_')
        if 'seed' in split[i + 1]:
            value = '_'.join(value_parts[:-1])
        else:
            value = value_parts[0]

        if value.isdigit():
            value = int(value)
        elif '.' in value:
            value = float(value)

        dict.update({key: value})

    return dict


def run_experiment(opts, model_path, config_name):
    """
    Prepares network, data and attacker, before starting the adversarial attack.

    Parameters
    ----------
    opts : dict
        Dictionary containing arguments of this script.
    model_path : pathlib.Path
        Path pointing to the file where trained model is stored.
    config_name : str
        String containing configuration of model under attack.
    """
    exp_dir = Path(RUN_DIR) / 'adversaries' / opts.name
    if not exp_dir.parent.exists():
        exp_dir.parent.mkdir()
    if not exp_dir.exists():
        exp_dir.mkdir()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_config = get_model_config(config_name)

    # get model
    print('Preparing model...')
    model = get_configured_model_architecture(model_config['arch'], device)
    load_model(model_path, model)

    # get data
    print('Preparing data...')
    emotion_dataset = EmotionDataset()
    train_set_size = int(len(emotion_dataset) * 0.8)
    test_set_size = int((len(emotion_dataset) - train_set_size) * 0.5)
    valid_set_size = (len(emotion_dataset) - train_set_size) - test_set_size
    seed = torch.Generator().manual_seed(42)    # always split data the same way
    train_set, valid_set, test_set = torch.utils.data.random_split(emotion_dataset,
                                                                   [train_set_size, valid_set_size, test_set_size],
                                                                   generator=seed)

    em_val_dataloader = DataLoader(valid_set, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers,
                                   drop_last=True, pin_memory=True, prefetch_factor=16)
    em_test_dataloader = DataLoader(test_set, batch_size=opts.batch_size, shuffle=False, num_workers=opts.num_workers,
                                    drop_last=True, pin_memory=True, prefetch_factor=16)

    # first, check validation and test metrics
    val_metrics = test(model, em_val_dataloader, nn.MSELoss().to(device), mets=['r2', 'corr_avg', 'rmse'])[0]
    test_metrics = test(model, em_test_dataloader, nn.MSELoss().to(device), mets=['r2', 'corr_avg', 'rmse'], phase='test')[0]
    print('Validation metrics: {}\nTest metrics: {}'.format(val_metrics, test_metrics))

    # set random seed
    set_random_seed(opts.seed)
    # load attack config and prepare attack, then start
    print('Preparing attack...')
    with open(Path(HOME_DIR) / 'adversary' / 'attack_config.yaml', 'r') as fp:
        config = yaml.load(fp, Loader=yaml.FullLoader)
    config.update({'architecture': model_config['arch'], 'model': model_path.parent.parent.name})
    attacker = Attacker(model=model, device=device, hparams=config, optimiser=get_optim_class(config['optimiser']),
                        log_dir=exp_dir, seed=opts.seed)
    print('Start attack now!')
    if opts.attack_test:
        attacker.attack(em_test_dataloader)
    else:
        attacker.attack(em_val_dataloader)


def get_paths(exp_dir: Path):
    """ Given directory of experiment, returns path were model was stored and directory name encoding according config file. """
    if not exp_dir.exists():
        raise NotADirectoryError('Please define a valid experiment directory, {} does not exist'.format(exp_dir))
    [model_path] = glob(str(exp_dir / 'saved_models' / '*.pt'))
    if not Path(model_path).exists():
        raise FileNotFoundError('No trained model found.')
    [config_path] = glob(str(exp_dir / '*=*'))
    if not Path(config_path).exists():
        raise NotADirectoryError('Directory containing config data was not found.')

    return Path(model_path), Path(config_path).name


def main():
    # parse arguments
    parser = opts_parser()
    opts = parser.parse_args()
    opts.batch_size = 1

    # get model save path
    model_path, config_name = get_paths(Path(opts.exp_dir))

    # run
    run_experiment(opts, model_path, config_name)


if __name__ == '__main__':
    main()
