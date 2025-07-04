import numpy as np
import pandas as pd

from train_models.utils import compute_metrics


def convert_strings_to_array(log_entries):
    """ Given log entries of arrays in string representation, creates numpy array containing values. """
    res_list = []
    for l in log_entries:
        splits = l.replace('\n', '').replace('[[', '').replace(']]', '').replace('  ', ' ').rsplit(' ')
        splits = np.array(list(filter(None, splits)), dtype=float)
        res_list.append(splits)
    return np.array(res_list)


def process_log_file(log_file):
    """ Reads given log file, and computes a set of different metrics of (before and after) attack. """
    log = pd.read_csv(log_file)

    # compute mean / median / std of db
    db_mean = np.mean(log['db'])
    db_median = np.median(log['db'])
    db_std = np.std(log['db'])

    # compute mean / median / std of norm
    norm_mean = np.mean(log['norm'])
    norm_median = np.median(log['norm'])
    norm_std = np.std(log['norm'])

    # compute mean / median / std of epochs
    epoch_mean = np.mean(log['epoch'])
    epoch_median = np.median(log['epoch'])
    epoch_std = np.std(log['epoch'])

    true_labels = convert_strings_to_array(log['true-label'])
    orig_preds = convert_strings_to_array(log['orig-pred'])
    new_preds = convert_strings_to_array(log['new-pred'])

    orig_metrics = compute_metrics(true_labels, orig_preds, ['corr_avg', 'r2', 'rmse', 'mse', 'corr', 'mae'], num_cols=8)
    new_metrics = compute_metrics(true_labels, new_preds, ['corr_avg', 'r2', 'rmse', 'mse', 'corr', 'mae'], num_cols=8)
    orig_metrics.update({'true_labels': true_labels})
    new_metrics.update({'true_labels': true_labels})
    orig_metrics.update({'preds': orig_preds})
    new_metrics.update({'preds': new_preds})

    return orig_metrics, new_metrics, [db_mean, db_median, db_std], \
           [norm_mean, norm_median, norm_std], [epoch_mean, epoch_median, epoch_std]
