import torch
import numpy as np
import torch.optim as optim

from tqdm import tqdm
from adversary.attacker import Attacker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# adversarial configuration (here with max. 50 iterations to compute perturbation)
config = {
    'alpha': None,
    'clip_eps': 0.001,
    'conv_method': 'corr',
    'lr': 0.002,
    'max_iters': 50,
    'mode': 'untargeted',
    'sign': True,
    'target': 'random',
    'every_n_epoch': 5
}


def adv_train(model, dataloader, optimizer, criterion, epoch, loss_weights, run_name):
    """
    Performs model updates for one (adversarial) training epoch.

    Parameters
    ----------
    model : torch.nn.Module
        Model that is (adversarially) trained.
    dataloader : torch.utils.data.DataLoader
        Dataloader containing data that is first optionally perturbed, and then used to update model parameters.
    optimizer : torch.optim.Optimizer
        Optimiser used to update parameters of model.
    criterion : torch.nn.Loss
        Loss function used to train the model.
    epoch : int
        Current epoch of model training, used to determine whether model parameters are also updated based on perturbed data or only on clean data.
    loss_weights : dict
        Dictionary containing weights for emotion loss ('output') and midlevel feature loss ('c_prob').
    run_name : str
        Name of this run to use for progress bar.

    Returns
    -------
    dict
        Containing average loss of this epoch (with key 'avg_loss').
    """
    attacker = Attacker(model=model, device=device, hparams=config, optimiser=optim.Adam)
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
            # we also have bottleneck predictions for midlevel features
            embedding = y['c_prob']
            ml_loss = criterion(embedding.float(), ml_targets.squeeze().float())
            # now weight + add up the losses
            # if loss_weights['c_prob'] == 0, we do not optimise for midlevel features despite having a bottleneck
            loss = loss_weights['output'] * loss + loss_weights['c_prob'] * ml_loss

        loss_list.append(loss.item())
        loss.backward()
        optimizer.step()

        # adversarial training part
        if epoch % config['every_n_epoch'] == 0:
            # first compute perturbations
            model.eval()
            adv_inputs = []
            for inp, lab in zip(inputs, labels):
                ad_ex, _, _, _, _, _, _ = attacker.compute_untargeted_adversary(inp.unsqueeze(0), lab.view(1, -1).float())
                adv_inputs.append(ad_ex.clone().detach().requires_grad_(False).to(device))
            adv_inputs = torch.cat(adv_inputs, dim=0)

            # then update model parameters based on predictions on adversarial examples
            model.train()
            optimizer.zero_grad()
            model.zero_grad()

            y = model(adv_inputs)
            output = y['output']
            loss = criterion(output.float(), labels.squeeze().float())
            loss.backward()
            optimizer.step()

        loss_list.append(loss.item())

    return {'avg_loss': np.mean(loss_list)}
