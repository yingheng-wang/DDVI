'''Exact ELBO of trained models on the test set (or on the validation set if the run used val_split):
    ELBO = E_q[log p(x|z)] + E_q[log p(z)] - E_q[log q(z|x)]   (semi-supervised: - KL(q(l|x) || p(l)) as well)
single-sample estimate per test example, averaged over the test set, in nats per image.
- log p(x|z): MNIST, Bernoulli decoder (minus the BCE summed over the 784 pixels). CIFAR-10, Gaussian decoder
  N(x; x_hat(z), s^2 I) with one variance per trained model, fitted by maximum likelihood on the training set
  (s^2 = mean per-pixel squared error under q).
- log p(z): the KDE prior density (priors.py), also used in training.
- log q: exact for AEVB (Gaussian), AEVB-IAF (IAF log-det) and DDVI (deterministic DDIM map, autograd 2x2 log-det);
  auxiliary-variable bound for H-AEVB (reverse model N(0, I) for z1); Girsanov path bound for PIS. Semi-supervised:
  q(l|x) q(z|x[, l]) with uniform p(l).
Usage: python evaluate_elbo.py RUN_DIR [RUN_DIR ...]   (writes RUN_DIR/elbo.json)'''
import json
import math
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

from ddvi.data import get_eval_data
from ddvi.models import build_model


def evaluate(run_dir):
    cfg = OmegaConf.load(os.path.join(run_dir, 'config.yaml'))
    torch.manual_seed(0)
    np.random.seed(0)
    model = build_model(cfg, training=False)
    model.load(os.path.join(run_dir, 'model.pt'))
    model.eval_mode()
    train_loader, test_loader, _, _ = get_eval_data(cfg)
    dev = model.device
    res = {}
    if not cfg.dataset.name.startswith('mnist'):
        se, n = 0.0, 0
        with torch.no_grad():
            for x in train_loader:
                x = x.to(dev)
                se += model.elbo_terms(x, need_q=False)[0].sum().item()
                n += x.size(0)
    R, LP, LQ, LK = [], [], [], []
    with torch.no_grad():
        for x in test_loader:
            r, lp, lq, lk = model.elbo_terms(x.to(dev))
            for lst, v in zip((R, LP, LQ, LK), (r, lp, lq, lk)):
                lst.append(v.cpu())
    R, LP, LQ, LK = (torch.cat(v) for v in (R, LP, LQ, LK))
    D = int(np.prod(x.shape[1:]))
    res.update(rec=R.mean().item(), log_p=LP.mean().item(), log_q=LQ.mean().item(), label_kl=LK.mean().item(),
               n_nonfinite=int((~torch.isfinite(LP - LQ)).sum()))
    if cfg.dataset.name.startswith('mnist'):
        log_lik = -res['rec']
    else:
        s2 = res['train_mse_per_pixel'] = se / (n * D)
        log_lik = -0.5 * D * math.log(2 * math.pi * s2) - res['rec'] / (2 * s2)
    res['log_lik'] = log_lik
    res['elbo'] = log_lik + res['log_p'] - res['log_q'] - res['label_kl']
    with open(os.path.join(run_dir, 'elbo.json'), 'w') as f:
        json.dump(res, f, indent=1)
    return res


if __name__ == '__main__':
    for d in sys.argv[1:]:
        cfg = OmegaConf.load(os.path.join(d, 'config.yaml'))
        if cfg.model.name.startswith('aaeb'):
            print(d, 'skipped (AAEB has no ELBO)')
            continue
        print(d, json.dumps(evaluate(d)), flush=True)
