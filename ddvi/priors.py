'''Structured 2-d priors p(z) (pinwheel, swiss roll, square) with 10 partitions each, and their densities.
The prior densities are approximated by Gaussian kernel density estimates fitted to prior samples: evaluate() for p(z)
(10,000 samples, mixture over five bandwidths, log-density clipped to [-1000, 1000]) and evaluate_partition() for the
class-conditional densities p(z|l) used in semi-supervised learning (2,000 samples per class, bandwidth 0.05, clipped
to [-300, 300]).'''
import math

import numpy as np
import torch

from .utils import get_device


class GaussianKDE:
    '''Gaussian kernel density estimate; returns log p(x) for each row of x.
    Follows GaussianKernel / KernelDensityEstimator of https://github.com/EugenHotaj/pytorch-generative (MIT).'''
    def __init__(self, train_xs, bandwidth):
        self.train_xs = train_xs
        self.bandwidth = bandwidth

    def __call__(self, xs):
        n, d = self.train_xs.shape
        n, h = torch.tensor(n, dtype=torch.float32), torch.tensor(self.bandwidth)
        Z = 0.5 * d * torch.log(2 * torch.tensor(np.pi)) + d * torch.log(h) + torch.log(n)
        diffs = (xs[:, None] - self.train_xs[None]) / h
        return torch.logsumexp(-0.5 * torch.norm(diffs, p=2, dim=-1) ** 2 - Z, dim=-1)


class BasePrior:
    n_classes = 10

    def __init__(self, latent_size):
        self.latent_size = latent_size
        self.device = get_device()
        self.kdes = []
        self.partition_kdes = None

    def sample(self, batch_size, g_classes=None, with_classes=False):
        '''Samples (and, if with_classes, their partition labels); g_classes fixes the partition of every sample.'''
        if g_classes is None:
            g_classes = np.random.randint(self.n_classes, size=batch_size)
        zs = self._sample(batch_size, g_classes)
        zs = torch.Tensor(zs).to(self.device)
        if with_classes:
            return zs, torch.Tensor(g_classes).long().to(self.device)
        return zs

    def evaluate(self, zs, nored=False):
        '''KDE log p(z) per sample (nored) or its mean.'''
        sigmas = [0.005, 0.008, 0.01, 0.03, 0.05]
        if not self.kdes:
            prior_samples = self.sample(10000)
            self.kdes = [GaussianKDE(prior_samples, s) for s in sigmas]
        scores = torch.stack([torch.clamp(kde(zs), -1000, 1000) for kde in self.kdes])
        out = torch.logsumexp(scores, 0) - math.log(len(sigmas))
        return out if nored else out.mean()

    def evaluate_partition(self, zs, partition):
        '''KDE log p(z|l = partition) per sample.'''
        if self.partition_kdes is None:
            self.partition_kdes = [GaussianKDE(self.sample(2000, g_classes=np.full(2000, i)), 0.05)
                                   for i in range(self.n_classes)]
        return torch.clamp(self.partition_kdes[partition](zs), -300, 300)


class PinWheelPrior(BasePrior):
    def __init__(self, latent_size, n_classes=10):
        super().__init__(latent_size)
        self.n_classes = n_classes
        self.radial_std = 0.15
        self.tangential_std = 0.005
        self.rate = 0.25

    def _sample(self, batch_size, g_classes):
        rads = np.linspace(0, 2 * np.pi, self.n_classes, endpoint=False)
        features = np.random.randn(batch_size, 2) * np.array([self.radial_std, self.tangential_std])
        features[:, 0] += 1.
        angles = rads[g_classes] + self.rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))
        return np.einsum('ti,tij->tj', features, rotations)


class SwissRollPrior(BasePrior):
    def __init__(self, latent_size, n_classes=10, noise_level=0):
        super().__init__(latent_size)
        self.n_classes = n_classes
        self.noise_level = noise_level

    def _sample(self, batch_size, g_classes):
        uni = np.random.uniform(0.0, 1.0, size=batch_size) / self.n_classes + g_classes / self.n_classes
        r = np.sqrt(uni) * 3.0
        rad = np.pi * 4.5 * np.sqrt(uni)
        zs = np.stack((r * np.cos(rad), r * np.sin(rad)), axis=-1)
        return zs + np.random.randn(batch_size, 2) * self.noise_level


class SquarePrior(BasePrior):
    def __init__(self, latent_size, n_classes=10, noise_level=0):
        super().__init__(latent_size)
        self.n_classes = n_classes
        self.noise_level = noise_level

    def _sample(self, batch_size, g_classes):
        pos = (g_classes + np.random.rand(batch_size)) / self.n_classes   # position along the perimeter
        zs = np.zeros((batch_size, 2))
        noises = np.random.randn(batch_size) / 50 * self.noise_level
        zs[pos < 1] = np.stack((np.ones_like(pos[pos < 1]) * -1 + noises[pos < 1], (pos[pos < 1] - 0.75) * 8 - 1), -1)
        zs[pos < 0.75] = np.stack(((2 - (pos[pos < 0.75] - 0.5) * 8) - 1, np.ones_like(pos[pos < 0.75]) * -1 + noises[pos < 0.75]), -1)
        zs[pos < 0.5] = np.stack((np.ones_like(pos[pos < 0.5]) * 1 + noises[pos < 0.5], (2 - (pos[pos < 0.5] - 0.25) * 8) - 1), -1)
        zs[pos < 0.25] = np.stack((pos[pos < 0.25] * 8 - 1, np.ones_like(pos[pos < 0.25]) * 1 + noises[pos < 0.25]), -1)
        return zs


class GaussPrior:
    '''N(0, I); used for the top-level latent of H-AEVB.'''
    def __init__(self, latent_size):
        self.latent_size = latent_size

    def evaluate(self, z, nored=False):
        out = -0.5 * self.latent_size * math.log(2 * math.pi) - 0.5 * torch.sum(z ** 2, -1)
        return out if nored else out.mean()


def get_prior(cfg):
    p, L, C = cfg.prior, cfg.model.latent_size, cfg.dataset.n_classes
    if p.type == 'pin_wheel':
        return PinWheelPrior(L, p.n_arc)
    if p.type == 'swiss_roll':
        return SwissRollPrior(L, C, noise_level=p.noise_level)
    if p.type == 'square':
        return SquarePrior(L, C, noise_level=p.noise_level)
    raise ValueError(p.type)
