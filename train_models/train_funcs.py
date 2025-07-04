import torch
import logging
import numpy as np

from tqdm import tqdm
from train_models.utils import compute_metrics

logger = logging.getLogger()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, dataloader, optimizer, criterion, epoch, loss_weights, run_name):
    """
    Performs model updates for one training epoch.

    Parameters
    ----------
    model: torch.nn.Module
        Model that is being trained.
    dataloader : torch.utils.data.DataLoader
        Dataloader containing data that is used to update model parameters.
    optimizer : torch.optim.Optimizer
        Optimiser used to update parameters of model.
    criterion : torch.nn.Loss
        Loss function used to train the model.
    epoch : int
        Current epoch of model training (unused here).
    loss_weights : dict
        Dictionary containing weights for emotion loss ('output') and midlevel feature loss ('c_prob').
    run_name : str
        Name of this run to use for progress bar.

    Returns
    -------
    dict
        Containing average loss of this epoch (with key 'avg_loss').
    """
    model.train()
    loss_list = []

    for batch_idx, batch in tqdm(enumerate(dataloader), ascii=False, total=len(dataloader), desc=run_name):
        song_ids, inputs, labels, ml_targets = batch
        inputs, labels, ml_targets = inputs.to(device), labels.to(device), ml_targets.to(device)
        inputs = inputs.unsqueeze(1)
        optimizer.zero_grad()
        y = model(inputs)
        output = y['output']
        loss = criterion(output.float(), labels.squeeze().float())
        if y['c_prob'] is not None:
            # this means we also have bottleneck predictions for midlevel features
            embedding = y['c_prob']
            ml_loss = criterion(embedding.float(), ml_targets.squeeze().float())
            # now weight + add up the losses
            # if loss_weights['c_prob'] == 0, we do not optimise for midlevel features despite having a bottleneck
            loss = loss_weights['output'] * loss + loss_weights['c_prob'] * ml_loss

        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()

    return {'avg_loss': np.mean(loss_list)}


def test(model, dataloader, criterion, epoch=-1, phase='val', **kwargs):
    """
    Computes validation / test performance of given model.

    Parameters
    ----------
    model: torch.nn.Module
        Model that is being tested.
    dataloader : torch.utils.data.DataLoader
        Dataloader containing data that is used to test model.
    criterion : torch.nn.Loss
        Loss function used to train and test the model.
    epoch : int
        Current epoch of model training (unused here).
    phase : str
        Describes phase of model training / data that performance is computed on (e.g., 'val', 'test'), used for logging.
    kwargs : dict
        Additional arguments.

    Returns
    -------
    return_dict : dict
        Dictionary containing average loss of emotion labels and predictions, and other metrics as defined with optional parameter mets = [...].
    labels_list : nd.array
        Stacked emotion labels for given dataset.
    preds_list : nd.array
        Stacked emotion predictions of model for given dataset.
    """

    mets = ['corr_avg'] if kwargs.get('mets') is None else kwargs['mets']
    test_output = 'output' if kwargs.get('test_output') is None else kwargs['test_output']

    model.eval()
    loss_list, preds_list, labels_list = [], [], []
    ml_pred_list, ml_targets_list = [], []
    return_dict = {}
    
    for batch_idx, batch in tqdm(enumerate(dataloader), ascii=True, total=len(dataloader), desc=f"Testing {test_output} ... "):
        _, inputs, labels, ml_targets = batch
        ml_targets_list.append(ml_targets.squeeze())        # collect midlevel targets
        inputs, labels, ml_targets = inputs.to(device), labels.to(device), ml_targets.to(device)
        if inputs.ndim < 4:
            inputs = inputs.unsqueeze(len(inputs.shape) - 2)
        output = model(inputs)
        emo_preds = output[test_output]     # collect emotion predictions
        if output['c_prob'] is not None:
            ml_pred_list.append(output['c_prob'].squeeze().cpu().detach().numpy())      # and, for bottleneck models, collect midlevel predictions

        loss = criterion(emo_preds.float(), labels.squeeze().float())   # compute loss on emotion labels

        # save emotion predictions, loss and emotion labels to according lists
        preds_list.append(emo_preds.cpu().detach().numpy())
        loss_list.append(loss.item())
        labels_list.append(labels.squeeze().cpu().detach().numpy())

    epoch_test_loss = np.mean(loss_list)
    # store average loss in return_dict
    return_dict['avg_loss'] = epoch_test_loss
    # finally, compute some metrics with the emotion and midlevel predictions, save them in return_dict
    if kwargs.get('compute_metrics', True) is True:
        emo_metrics = compute_metrics(np.vstack(labels_list), np.vstack(preds_list), metrics_list=mets,
                                      num_cols=kwargs.get('compute_metrics', 8))
        return_dict.update({f'{phase}_emo_'+k: v for k, v in emo_metrics.items()})
        if output['c_prob'] is not None:
            ml_metrics = compute_metrics(np.vstack(ml_targets_list), np.vstack(ml_pred_list), metrics_list=mets)
            return_dict.update({f'{phase}_ml_'+k: v for k, v in ml_metrics.items()})

    return return_dict, np.vstack(labels_list), np.vstack(preds_list)
