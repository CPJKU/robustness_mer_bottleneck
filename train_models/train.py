import time
import torch
import hashlib
import logging
import argparse
import torch.utils.data as torchdata

from pathlib import Path
from torch import optim, nn
from datetime import datetime as dt
from torch.utils.tensorboard import SummaryWriter

from paths import RUN_DIR
from data.emotion import EmotionDataset
from train_models.train_funcs import train, test
from train_models.adv_train_funcs import adv_train
from train_models.vggstyle import Audio2Target, Audio2Ml2Emo
from train_models.utils import dset_to_loader, EarlyStopping, generate_run_name, load_model


def opts_parser():
    """ Command line argument setup. """
    parser = argparse.ArgumentParser(description='Script to train model on emotion recognition.')
    parser.add_argument('--arch', type=str, choices=['a2e', 'a2m2e'],
                        help='Type of model architecture to be trained, either "a2e" (standard) or "a2m2e" (bottleneck).')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate to start training with.')
    parser.add_argument('--task_loss_weight', type=float, default=0.5,
                        help='Factor weighting loss during training, with loss = emotion_loss x weight + midlevel_loss x (1- weight).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed used during training.')
    parser.add_argument('--adversarial', action='store_true',
                        help='If true, do adversarial training instead of regular training.')
    return parser


def set_random_seed(seed):
    # make everything deterministic
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def get_data_loaders(rand_seed):
    """ Creates data splits of emotion dataset, sets random seed and returns dataloaders for training, validation and testing. """
    emotion_dataset = EmotionDataset()
    train_set_size = int(len(emotion_dataset) * 0.8)  # 80% for training
    test_set_size = int((len(emotion_dataset) - train_set_size) * 0.5)  # 10% for test and validation
    valid_set_size = (len(emotion_dataset) - train_set_size) - test_set_size
    seed = torch.Generator().manual_seed(42)  # fix as we want same split across all random seeds
    train_set, valid_set, test_set = torchdata.random_split(emotion_dataset,
                                                            [train_set_size, valid_set_size, test_set_size],
                                                            generator=seed)
    set_random_seed(rand_seed)
    em_tr_dataloader = dset_to_loader(train_set, bs=8, shuffle=True)
    em_val_dataloader = dset_to_loader(valid_set, bs=8, shuffle=False)
    em_test_dataloader = dset_to_loader(test_set, bs=8, shuffle=False)
    return em_test_dataloader, em_tr_dataloader, em_val_dataloader


def get_configured_model_architecture(arch_name, device):
    """ Given architecture name and device, returns according model (loaded to device). """
    if arch_name == 'a2e':
        model = Audio2Target(num_targets=8).to(device)
    elif arch_name == 'a2m2e':
        model = Audio2Ml2Emo(num_targets=8).to(device)
    else:
        raise ValueError('architecture {} is not implemented (yet).'.format(arch_name))
    return model


def prep_loggers(run_dir, args, loss_weights, run_name):
    """ Prepares terminal- and tensorboard logger. """
    logger = logging.getLogger()
    fh = logging.FileHandler(Path(run_dir) / f'{run_name}.log')
    sh = logging.StreamHandler()  # for printing to terminal or console
    formatter = logging.Formatter('%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    logger.setLevel(logging.INFO)

    tb_logger = SummaryWriter(Path(run_dir) / generate_run_name(args))
    tb_logger.add_hparams(vars(args), {})
    tb_logger.add_hparams(loss_weights, {})

    return logger, tb_logger


def run(args):
    """ Method that setups everything required for training, runs training procedure, and finishes up with some logging. """
    # prepare arguments, run directories...
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_weights = {'output': args.task_loss_weight, 'c_prob': 1 - args.task_loss_weight}

    dtstr = dt.now().strftime("%Y-%m-%d_%H-%M-%S")
    name_hash = hashlib.sha1()
    name_hash.update(str(time.time()).encode('utf-8'))
    run_hash = name_hash.hexdigest()[:5]
    run_name = f'{run_hash}_{dtstr}'
    run_dir = Path(RUN_DIR) / 'cem' / str(args.seed) / run_name
    run_dir.mkdir(parents=True)

    # prepare logging
    logger, tb_logger = prep_loggers(run_dir, args, loss_weights, run_name)
    # get data loaders
    em_test_dataloader, em_tr_dataloader, em_val_dataloader = get_data_loaders(args.seed)
    # get model
    model = get_configured_model_architecture(args.arch, device)
    # prepare early stopping, optimiser and loss function
    es = EarlyStopping(patience=50, condition="maximize", verbose=True,
                       save_dir=Path(run_dir) / 'saved_models', saved_model_name=f"cem_{args.arch}_{run_hash}")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss().to(device)

    # start training
    for epoch in range(1, 200):
        if args.adversarial:
            tr_losses = adv_train(model, em_tr_dataloader, optimizer, criterion, epoch, loss_weights,
                                  run_hash + f'_epoch-{epoch}')
        else:
            tr_losses = train(model, em_tr_dataloader, optimizer, criterion, epoch, loss_weights,
                              run_hash + f'_epoch-{epoch}')
        val_metrics = test(model, em_val_dataloader, criterion, mets=['r2', 'corr_avg', 'rmse'])[0]

        # log
        logger.info(f"tr_loss: {tr_losses['avg_loss']}")
        tb_logger.add_scalar('loss/train', tr_losses['avg_loss'], epoch)
        tb_logger.add_scalar('loss/valid', val_metrics['avg_loss'], epoch)
        logger.info(' '.join([f"{k}={round(v, 4)}" for k, v in val_metrics.items()]))

        if 'val_ml_corr_avg' in val_metrics.keys():
            es(val_metrics['val_emo_corr_avg'] + val_metrics['val_ml_corr_avg'], model)
        else:
            es(val_metrics['val_emo_corr_avg'], model)
        if es.early_stop:       # here we want to do early stopping
            logger.info(f"Early stop - trained for {epoch - es.counter} epochs - best metric {es.best_score}")
            break

    log_final_metrics(es.save_path, epoch, criterion, em_tr_dataloader, em_val_dataloader, em_test_dataloader, model, tb_logger)


def log_final_metrics(save_path, epoch, criterion, train_loader, valid_loader, test_loader, model, tb_logger):
    """ Loads model saved after training, computes final training / validation / test metrics and logs them. """
    load_model(save_path, model)
    train_metrics = test(model, train_loader, criterion, epoch, mets=['r2', 'corr_avg', 'rmse', 'mse'], phase='train')[0]
    val_metrics = test(model, valid_loader, criterion, epoch, mets=['r2', 'corr_avg', 'rmse', 'mse'], phase='val')[0]
    test_metrics = test(model, test_loader, criterion, epoch, mets=['r2', 'corr_avg', 'rmse', 'mse'], phase='test')[0]
    log_dict(tb_logger, train_metrics, 'final_train', epoch)
    log_dict(tb_logger, val_metrics, 'final_valid', epoch)
    log_dict(tb_logger, test_metrics, 'final_test', epoch)


def log_dict(tb_logger, metrics, phase, epoch):
    """ Given tensorboard logger, logs all scalars in metrics-dict for defined phase. """
    for k, v in metrics.items():
        tb_logger.add_scalar('{}/{}'.format(k, phase), v, epoch)


def main():
    parser = opts_parser()
    args = parser.parse_args()

    run(args)


if __name__ == '__main__':
    main()
