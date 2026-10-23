import math
import random

import numpy as np
import torch


def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def sample(mu, log_var, ret_log_prob=False):
    '''Reparameterized sample from N(mu, diag exp(log_var)), optionally with its log-density.'''
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    if ret_log_prob:
        return mu + eps * std, -torch.sum(0.5 * log_var + (eps ** 2) / 2 + 0.5 * math.log(2 * math.pi), -1)
    return mu + eps * std


def normal_logpdf(x, mu, log_var):
    return -0.5 * torch.sum(math.log(2 * math.pi) + log_var + (x - mu) ** 2 / log_var.exp(), -1)


def kl_to_standard_normal(mu, log_var):
    '''KL(N(mu, diag exp(log_var)) || N(0, I)) per sample.'''
    return -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), -1)


def frange_cycle_linear(n_iter, start=0.0, stop=1.0, n_cycle=4, ratio=0.5):
    L = np.ones(n_iter) * stop
    period = n_iter / n_cycle
    step = (stop - start) / (period * ratio)
    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i + c * period) < n_iter):
            L[int(i + c * period)] = v
            v += step
            i += 1
    return L


def get_kld_weights(kld_weight, num_epochs, schedule):
    '''Per-epoch weight of the regularization term: 'constant', or 'linear' (warm-up from 0 over the first 80% of the
    epochs, then constant).'''
    if schedule == 'constant':
        return np.full(num_epochs, float(kld_weight))
    if schedule == 'linear':
        n = int(num_epochs * 0.8)
        if n == 0:
            return np.full(num_epochs, float(kld_weight))
        return np.concatenate((frange_cycle_linear(n, start=0, stop=kld_weight, n_cycle=1, ratio=1),
                               np.full(num_epochs - n, kld_weight)))
    raise ValueError(schedule)


def cs_entropy(cs):
    '''Mean entropy of the class probabilities cs (one row per example).'''
    if len(cs) == 0:
        return torch.zeros(1, device=cs.device)
    return -torch.sum((cs + 1e-6) * torch.log(cs + 1e-6)) / cs.size(0)


def label_kl(logits):
    '''q(l|x) and KL(q(l|x) || uniform) per sample.'''
    q = torch.softmax(logits, -1)
    return q, torch.sum(q * (torch.log(q + 1e-8) + math.log(q.size(1))), -1)


def flow_logdet(flow, y):
    '''z = flow(y) for y of shape (B, 2) with independent rows; returns z and log |det dz/dy| per row (autograd).'''
    y = y.detach().requires_grad_(True)
    with torch.enable_grad():
        z = flow(y)
        g0 = torch.autograd.grad(z[:, 0].sum(), y, retain_graph=True)[0]
        g1 = torch.autograd.grad(z[:, 1].sum(), y)[0]
    det = g0[:, 0] * g1[:, 1] - g0[:, 1] * g1[:, 0]
    return z.detach(), torch.log(det.abs() + 1e-30)
