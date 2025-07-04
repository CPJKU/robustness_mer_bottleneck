import torch
import numpy as np

from pathlib import Path
from sklearn import metrics
from paths import RUN_DIR
from scipy.stats import pearsonr
from torch.utils.data import DataLoader


def generate_run_name(args, exclude_args=None):
    """ Creates a string containing all given arguments (except excluded ones) to name run-directory. """
    # Initialize an empty list to store argument-value pairs
    arg_pairs = []
    if exclude_args is None:
        exclude_args = []

    # Iterate through parsed arguments and add them to arg_pairs
    for arg in vars(args):
        if arg not in exclude_args:
            arg_value = getattr(args, arg)
            arg_pairs.append(f'{arg}={arg_value}')

    # Join the argument-value pairs with '_' to create the run name
    run_name = '_'.join(arg_pairs)
    return run_name


def dset_to_loader(dset, bs, num_workers=8, shuffle=False):
    """ Creates dataloader from dataset with given batch-size, number of workers, and shuffled or not. """
    return DataLoader(dset, batch_size=bs, shuffle=shuffle, num_workers=num_workers, drop_last=True, pin_memory=False)


def load_model(model_weights_path, model):
    """ Loads stored weights from given path to given model, and sets mode to eval. """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print('Loaded model to {}...'.format(device))
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(model_weights_path))
    else:
        model.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
    model.eval()


def compute_metrics(y, y_hat, metrics_list, **kwargs):
    """ Function that computes various metrics (given on list) for the defined labels y and predictions y_hat. """
    metrics_res = {}
    for metric in metrics_list:
        Y, Y_hat = y, y_hat
        if metric in ['rocauc-macro', 'rocauc']:
            metrics_res[metric] = metrics.roc_auc_score(Y, Y_hat, average='macro')
        if metric == 'rocauc-micro':
            metrics_res[metric] = metrics.roc_auc_score(Y, Y_hat, average='micro')
        if metric in ['prauc-macro', 'prauc']:
            metrics_res[metric] = metrics.average_precision_score(Y, Y_hat, average='macro')
        if metric == 'prauc-micro':
            metrics_res[metric] = metrics.average_precision_score(Y, Y_hat, average='micro')

        if metric == 'corr_avg':
            corr, pval = [], []
            for i in range(kwargs.get("num_cols", 7)):
                c, p = pearsonr(Y[:, i], Y_hat[:, i])
                corr.append(c)
            metrics_res['corr_avg'] = np.mean(corr)

        if metric == 'corr':
            corr, pval = [], []
            for i in range(kwargs.get("num_cols", 7)):
                c, p = pearsonr(Y[:, i], Y_hat[:, i])
                corr.append(c)
            metrics_res['corr'] = corr

        if metric == 'mae':
            metrics_res[metric] = metrics.mean_absolute_error(Y, Y_hat)
        if metric == 'r2':
            metrics_res[metric] = metrics.r2_score(Y, Y_hat)
        if metric == 'r2_raw':
            metrics_res[metric] = metrics.r2_score(Y, Y_hat, multioutput='raw_values')
        if metric == 'mse':
            metrics_res[metric] = metrics.mean_squared_error(Y, Y_hat)
        if metric == 'rmse':
            metrics_res[metric] = np.sqrt(metrics.mean_squared_error(Y, Y_hat))
        if metric == 'rmse_raw':
            metrics_res[metric] = np.sqrt(metrics.mean_squared_error(Y, Y_hat, multioutput='raw_values'))

    return metrics_res


class EarlyStopping:
    """ Implementation of early stopping, which keeps track of best value of a metric, to stop training if value does not improve after given patience. """

    def __init__(self, patience=7, verbose=False, delta=0, save_dir=RUN_DIR + 'cem',
                 saved_model_name='model_chkpt', condition='minimize'):
        """
        Parameters
        ----------
        patience : int
            How long to wait for improvement of metric before stopping.
        verbose : bool
            If True, prints message for each improvement.
        delta : float
            Minimum change in the monitored metric to qualify as an improvement.
        save_dir : str
            Directory to which model should be saved after no more improvement is made.
        saved_model_name : str
            Name with which model checkpoint should be saved.
        condition : str
            Either 'maximize' or 'minimize', defines whether given metric should be maximized or minimized.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.save_dir = Path(save_dir)
        if not self.save_dir.exists():
            self.save_dir.mkdir(parents=True)
        self.saved_model_name = saved_model_name
        self.save_path = self.save_dir / (self.saved_model_name + '.pt')
        if condition not in ['minimize', 'maximize']:
            raise ValueError('Condition must be either minimize or maximize.')
        self.condition = condition
        self.metric_best = np.Inf if condition == 'minimize' else -np.Inf

    def __call__(self, metric, model):
        """
        Monitors current metric of model training, and checks whether improvement was made or not.
        Parameters
        ----------
        metric : float
            Metric which is monitored for early stopping. Should be either maximised or minimised.
        model : torch.nn.Module
            Model which is currently trained, necessary for saving if new best score is detected.
        """

        score = metric if self.condition == 'maximize' else -metric

        if self.best_score is None:
            # nothing has been observed yet
            self.best_score = score
            self.save_checkpoint(metric, model)
        elif score < self.best_score + self.delta:
            # no improvement was observed, increase counter
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                # we reached the patience, so trigger early stopping
                self.early_stop = True
        else:
            # there was an improvement, so reset counter and update best found score, save model checkpoint
            self.best_score = score
            self.save_checkpoint(metric, model)
            self.counter = 0

    def save_checkpoint(self, metric, model):
        """
        Saves given model checkpoint to predefined directory.
        Parameters
        ----------
        metric : float
            Metric which is monitored during early stopping process. Necessary for verbose mode.
        model : torch.nn.Module
            Model that should be saved.
        """
        if self.verbose:
            print(
                f'Metric improved ({self.condition}) ({self.metric_best:.6f} --> {metric:.6f}).  Saving model to {str(self.save_dir / (self.saved_model_name + ".pt"))}')
        torch.save(model.state_dict(), self.save_path)
        self.metric_best = metric
