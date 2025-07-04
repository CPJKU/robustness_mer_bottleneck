import yaml
import torch
import librosa
import numpy as np
import torch.nn.functional as F

from tqdm import tqdm
from pathlib import Path
from sklearn import metrics
from math import ceil, floor
from scipy.stats import pearsonr

from adversary.csv_logger import Logger

SR = 22050


def mel_spec_snr(x, delta):
    """ Approximates SNR (of Mel-spectrograms). """
    ampl_x = librosa.feature.inverse.mel_to_stft(x.squeeze(), sr=SR)
    ampl_delta = librosa.feature.inverse.mel_to_stft(delta.squeeze(), sr=SR)

    db_x = librosa.amplitude_to_db(ampl_x)
    db_delta = librosa.amplitude_to_db(ampl_delta)

    snr = db_x - db_delta
    return np.mean(snr)


class Attacker:
    """ Attacker class, wrapping adversarial updates and saving final perturbations. """
    def __init__(self, model, device, optimiser, hparams, log_dir=None, seed=None):
        """
        Parameters
        ----------
        model : torch.nn.Module
            Model for which adversarial perturbations are computed.
        device : torch.device
            Device of model (and data).
        optimiser : torch.optim.Optimizer
            Optimiser used to compute adversarial perturbations.
        hparams : dict
            Dictionary containing hyperparameters of attack.
        log_dir : str or Path, optional
            Directory where we want to store adversaries, attack-hyperparameters, and the csv-logging. If None, nothing is stored.
        seed : int, optional
            Random seed.
        """
        self.model = model
        self.model.eval()       # we are not updating the model itself here
        self.model = self.model.to(device)
        self.device = device
        self.optimiser = optimiser

        self.mode = hparams['mode']
        self.target = hparams['target']
        self.lr = hparams['lr']
        self.max_it = hparams['max_iters']
        self.sign = hparams['sign']
        self.clip_eps = hparams['clip_eps']
        self.alpha = hparams['alpha']
        self.conv_method = hparams['conv_method']

        if log_dir:
            self.log_dir = Path(log_dir)
            columns = ['file', 'epoch', 'db', 'norm', 'converged', 'true-label', 'orig-pred', 'new-pred', 'adv-target']
            self.logger = Logger(self.log_dir / 'log.csv', columns=columns)
            self.seed = seed
            self.log_hparams(hparams)
        else:
            self.log_dir = None
            self.logger = None

    def log_hparams(self, hparams):
        """ Dumps hyperparameters of experiment into yaml-file. """
        with open(self.log_dir / 'hparams.yaml', 'w') as fp:
            yaml.dump(hparams, fp)

    def optimiser_step(self, delta, optimiser, ascent):
        """ Performs one update step with optimiser. """
        if self.sign:
            delta.grad = torch.sign(delta.grad)
        if ascent:      # for untargeted attack, we want to do gradient ascent
            delta.grad *= -1.
        optimiser.step()

        with torch.no_grad():
            if self.clip_eps:       # clip to restrict perturbation within epsilon ball
                delta.clamp(min=-self.clip_eps, max=self.clip_eps)

    def update(self, pred, delta, optimiser, adv_targets=None, labels=None):
        """ Updates adversarial perturbation (in-place), either moving towards targets (targeted), or moving away from labels (untargeted). """
        optimiser.zero_grad()
        if adv_targets is not None:
            # targeted attack
            if self.alpha:
                loss = self.alpha * F.mse_loss(pred, adv_targets) + torch.sum(delta ** 2)
            else:
                loss = F.mse_loss(pred, adv_targets)
            loss.backward()
            self.optimiser_step(delta, optimiser, False)
        elif labels is not None:
            # untargeted attack
            if self.alpha:
                loss1 = self.alpha * F.mse_loss(pred, labels)
                loss1.backward()
                self.optimiser_step(delta, optimiser, True)
                optimiser.zero_grad()
                loss2 = torch.sum(delta ** 2)
                loss2.backward()
                self.optimiser_step(delta, optimiser, False)
            else:
                loss = F.mse_loss(pred, labels)
                loss.backward()
                self.optimiser_step(delta, optimiser, True)
        else:
            raise AttributeError('Either targets or labels must be given for update.')

    def compute_untargeted_adversary(self, input, labels, compute_snr=True):
        """ Computes perturbation for an input (and ground-truth labels) iteratively for untargeted attack. """
        # initialise delta (perturbation) and optimiser
        delta = torch.zeros_like(input).to(self.device)
        delta.requires_grad = True
        optimiser = self.optimiser([delta], lr=self.lr)
        best_delta = delta.detach().cpu().clone()

        cur_target_threshold = None
        conv = False

        it = 1
        new_pred = None
        for it in range(1, self.max_it + 1):
            new_pred = self.model(input + delta)['output']
            # get performance metrics (MSE and correlation)
            mse = metrics.mean_squared_error(labels.cpu(), new_pred.detach().cpu())
            corr, _ = pearsonr(labels.cpu().squeeze(), new_pred.detach().cpu().squeeze())

            if not cur_target_threshold:
                # first, define initial threshold
                cur_target_threshold = floor(mse * 10) / 10 if self.conv_method == 'mse' else ceil(corr * 10) / 10

            # check whether we're already far away enough
            if (self.conv_method == 'mse' and mse >= cur_target_threshold) \
                    or (self.conv_method == 'corr' and corr <= cur_target_threshold):
                best_delta = delta.detach().cpu().clone()
                if self.conv_method == 'corr' and cur_target_threshold >= -0.9:
                    cur_target_threshold -= 0.1
                elif self.conv_method == 'mse' and cur_target_threshold <= 0.9:
                    cur_target_threshold += 0.1
                else:
                    print('Minimal threshold achieved ({}: {}), so we stop updating delta...'.format(
                        self.conv_method, cur_target_threshold
                    ))
                    conv = True
                    break

            # otherwise, update delta (in-place)
            self.update(new_pred, delta, optimiser, adv_targets=None, labels=labels)
            # output some info
            print('\rep %s/%s; mse: %.1f, corr: %.1f, cur_th: %.1f' % (it, self.max_it, mse, corr, cur_target_threshold),
                  flush=True, end='')

        # compute norm / SNR of delta, store it all
        snr = mel_spec_snr(input.cpu().numpy(), best_delta.cpu().detach().numpy()) if compute_snr else None
        norm = torch.sqrt((best_delta ** 2).sum()).detach().cpu().item()

        return (input.cpu() + best_delta).detach(), snr, norm, it, new_pred.detach().cpu(), conv, cur_target_threshold

    def attack(self, data):
        """ Outer attack loop that iterates over data, and calls computation of adversaries. """
        if self.log_dir and not (self.log_dir / 'specs').exists():
            # create subdirectory for saving adversarial examples
            (self.log_dir / 'specs').mkdir()

        for batch_idx, batch in tqdm(enumerate(data)):
            song_ids, inputs, labels, ml_targets = batch
            inputs, labels, ml_targets = inputs.to(self.device), labels.to(self.device), ml_targets.to(self.device)
            inputs = inputs.unsqueeze(1)
            labels = labels.view(1, -1).float()

            with torch.no_grad():
                orig_pred = self.model(inputs)['output']
            if self.mode == 'untargeted':
                ad_ex, snr, norm, eps, new_pred, conv, adv_target = self.compute_untargeted_adversary(inputs, labels)
            else:
                adv_target = None
                raise NotImplementedError('Only untargeted attack is supported (make sure to update attack configuration)!')

            # log result, store adversaries
            if self.logger:
                self.logger.append([Path(song_ids[0]).name, eps, snr, norm, conv, labels.cpu().numpy(), orig_pred.cpu().numpy(),
                                    new_pred.cpu().numpy(), adv_target if adv_target else None])
                self.save_adversary(Path(song_ids[0]).name, ad_ex)

    def save_adversary(self, filename, ad_ex):
        """
        Given filename and adversarial example, stores it.

        Parameters
        ----------
        filename : str or Path
            (Original) file name of adversarial example to be stored.
        ad_ex : torch.Tensor
            Adversarial example to be stored.
        """
        save_path = (self.log_dir / 'specs' / filename).with_suffix('.npy')
        spec = ad_ex.detach().cpu().numpy()
        np.save(save_path, spec)
