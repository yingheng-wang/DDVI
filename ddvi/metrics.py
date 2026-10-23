'''Latent-space metrics reported in the paper: KNN accuracy, latent NLL and MMD.'''
import json
import logging
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import colors
from scipy.special import logsumexp
from sklearn.neighbors import KernelDensity, KNeighborsClassifier

from .data import get_eval_data

log = logging.getLogger(__name__)


def knn_accuracy(train_latents, train_labels, test_latents, test_labels, k=20):
    '''Accuracy of a k-nearest-neighbor classifier fitted on the training latents.'''
    return float(KNeighborsClassifier(n_neighbors=k).fit(train_latents, train_labels).score(test_latents, test_labels))


def latent_nll(latents, prior_samples):
    '''-E_{z ~ p(z)}[log q(z)], with the aggregate posterior q(z) a Gaussian KDE of the test latents (mixture over five
    bandwidths); measures how well the latents cover the prior.'''
    sigmas = [0.05, 0.8, 0.1, 0.3, 0.5]
    scores = [KernelDensity(kernel='gaussian', bandwidth=s).fit(latents).score_samples(prior_samples) for s in sigmas]
    return float(-np.mean(logsumexp(np.asarray(scores), 0) - np.log(len(sigmas))))


def mmd(x, gen_x, sigma=(2, 5, 10, 20, 40, 80)):
    '''Multi-bandwidth RBF-kernel MMD between data x and generated samples gen_x (flattened).'''
    X = torch.cat([gen_x, x], 0)
    XX = torch.matmul(X, X.T)
    X2 = torch.sum(X * X, 1, keepdims=True)
    exponent = XX - 0.5 * X2 - 0.5 * X2.T
    s = torch.cat([torch.full((gen_x.shape[0], 1), 1.0 / gen_x.shape[0]),
                   -torch.full((x.shape[0], 1), 1.0 / x.shape[0])], 0)
    S = torch.matmul(s, s.T)
    loss = 0
    for sg in sigma:
        loss += torch.sum(S * torch.exp(1.0 / sg * exponent))
    return float(torch.sqrt(loss))


def plot_latents(latents, path, labels=None):
    plt.clf()
    cmap = colors.ListedColormap(['tab:blue', 'tab:green', 'tab:red', 'tab:cyan', 'tab:purple', 'tab:olive', 'black',
                                  'tab:pink', 'tab:gray', 'tab:brown'])
    plt.scatter(latents[:, 0], latents[:, 1], s=5, alpha=0.8, c=labels, cmap=cmap if labels is not None else None)
    plt.savefig(path, bbox_inches='tight')


def evaluate_metrics(model, out_dir=None):
    '''KNN-20 accuracy, latent NLL and MMD of a trained model on the test (or validation) set; writes metrics.json,
    latent.npy (test latents) and latent.png to out_dir.'''
    cfg = model.cfg
    train_loader, test_loader, train_labels, test_labels = get_eval_data(cfg)
    latents = model.encode(test_loader)
    train_latents = model.encode(train_loader)
    res = {'acc': knn_accuracy(train_latents, train_labels, latents, test_labels),
           'latent_nll': latent_nll(latents, model.prior.sample(10000).cpu().numpy())}

    prior_samples = model.prior.sample(10000)
    model.P.eval()
    with torch.no_grad():
        gen = torch.cat([model.P(prior_samples[i:i + model.batch_size]).cpu()
                         for i in range(0, 10000, model.batch_size)], 0)
    test_x = torch.cat([x.cpu() for x in test_loader], 0)
    res['mmd'] = mmd(test_x.view(test_x.size(0), -1), gen.view(gen.size(0), -1))
    log.info(f'Metrics: {res}')
    if out_dir is not None:
        np.save(os.path.join(out_dir, 'latent.npy'), latents)
        plot_latents(latents, os.path.join(out_dir, 'latent.png'), test_labels)
        with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
            json.dump(res, f, indent=1)
    return res
