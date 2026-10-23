import logging
import time

import numpy as np
import torch

from ..priors import get_prior
from ..utils import get_device, get_kld_weights

log = logging.getLogger(__name__)


class BaseModel:
    '''Common setup and training loop. Subclasses define build(), _train_epoch(), encode() and, for the exact ELBO
    (evaluate_elbo.py), elbo_terms(). ckpt_modules lists the networks stored in model.pt.
    training=False builds the networks only (no prior-denoiser warm-up or warm start), for loading a checkpoint.'''
    ckpt_modules = ('Q', 'P')

    def __init__(self, cfg, training=True):
        m = cfg.model
        self.cfg = cfg
        self.training = training
        self.latent_size = m.latent_size
        self.num_epochs = m.num_epochs
        self.hidden_size = m.hidden_size
        self.lr = m.lr
        self.batch_size = m.batch_size
        self.save_folder = cfg.save_folder
        self.device = get_device()
        self.prior = get_prior(cfg)
        self.mnist = cfg.dataset.name.startswith('mnist')
        self.n_classes = cfg.dataset.n_classes
        # MNIST: Bernoulli decoder (BCE); CIFAR-10: Gaussian decoder (squared error); averaged over pixels in training
        if self.mnist:
            self.criterion, self.criterion_nored = torch.nn.BCELoss(), torch.nn.BCELoss(reduction='none')
        else:
            self.criterion, self.criterion_nored = torch.nn.MSELoss(), torch.nn.MSELoss(reduction='none')
        if 'kld_weight' in m:
            self.kld_weights = get_kld_weights(m.kld_weight, self.num_epochs, m.kld_schedule)
        self.ep = 0
        self.build()

    def build(self):
        raise NotImplementedError

    def _call_schedulers(self):
        pass

    def train(self, dataloader, callback=None):
        log.info('Start training')
        start = time.time()
        for epoch in range(self.num_epochs):
            self.ep = epoch
            st = time.time()
            res = self._train_epoch(dataloader)
            self._call_schedulers()
            if epoch % 5 == 0 or epoch == self.num_epochs - 1:
                log.info(f'Epoch {epoch}: losses {np.round(res, 4)}, {(time.time() - st):.1f} s/epoch, '
                         f'{(time.time() - start) / 60:.1f} min in total')
            if callback is not None:
                callback(self, epoch)
        log.info(f'End training - training time: {time.time() - start:.0f} seconds')

    def rec_sum(self, x_hat, x):
        '''Reconstruction loss summed over pixels, per example: -log p(x|z) for MNIST (Bernoulli), the squared error for
        CIFAR-10 (the Gaussian log-likelihood is formed from it in evaluate_elbo.py).'''
        return self.criterion_nored(x_hat, x).reshape(x.size(0), -1).sum(-1)

    def elbo_terms(self, x, need_q=True):
        '''Per-example (rec_sum, log p(z), log q(z|x), KL(q(l|x) || p(l))) for one posterior sample; with
        need_q=False only rec_sum is computed.'''
        raise NotImplementedError(f'{type(self).__name__} has no ELBO')

    @torch.no_grad()
    def encode(self, dataloader):
        '''Latents of all examples of dataloader (numpy array).'''
        self.eval_mode()
        return np.concatenate([self._encode(x.to(self.device)).cpu().numpy() for x in dataloader], 0)

    def eval_mode(self):
        for n in self.ckpt_modules:
            getattr(self, n).eval()

    def _extra_state(self):
        return {}

    def _load_extra_state(self, ck):
        pass

    def save(self, path):
        ck = {n: getattr(self, n).state_dict() for n in self.ckpt_modules}
        ck.update(self._extra_state())
        torch.save(ck, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.device)
        for n in self.ckpt_modules:
            getattr(self, n).load_state_dict(ck[n])
        self._load_extra_state(ck)
