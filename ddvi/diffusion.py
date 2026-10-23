'''Gaussian diffusion with an x-independent noise predictor, used by the simplified DDVI (Section 3.4 / Appendix H):
linear beta schedule from 1e-4 to 0.02, deterministic DDIM reverse process y_T -> z, Huber noise-prediction loss.'''
import logging

import torch
import torch.nn.functional as F
from torch import nn, optim

from .nets import ConditionalModel

log = logging.getLogger(__name__)


def get_by_idx(vals, idx, x_shape):
    out = vals.gather(-1, idx.cpu())
    return out.reshape(idx.shape[0], *((1,) * (len(x_shape) - 1))).to(idx.device)


class GaussianDiffusionNoise(nn.Module):
    def __init__(self, model, timesteps, latent_size, device='cpu'):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.latent_size = latent_size
        self.device = device
        self.betas = torch.linspace(1e-4, 0.02, timesteps)
        self.alphas = 1 - self.betas
        bar_alphas = torch.cumprod(self.alphas, axis=0)
        self.sqrt_bar_alphas = torch.sqrt(bar_alphas)
        self.sqrt_one_minus_bar_alphas = torch.sqrt(1. - bar_alphas)
        self.sqrt_one_minus_bar_alphas_prev = F.pad(self.sqrt_one_minus_bar_alphas[:-1], (1, 0), value=0.0)

    def q_sample(self, x0, t, noise):
        return (get_by_idx(self.sqrt_bar_alphas, t, x0.shape) * x0
                + get_by_idx(self.sqrt_one_minus_bar_alphas, t, x0.shape) * noise)

    def p_sample(self, xt, t):
        '''One deterministic DDIM step y_t -> y_{t-1}.'''
        sqrt_one_minus_bar_alphas_t = get_by_idx(self.sqrt_one_minus_bar_alphas, t, xt.shape)
        sqrt_one_minus_bar_alphas_prev_t = get_by_idx(self.sqrt_one_minus_bar_alphas_prev, t, xt.shape)
        sqrt_recip_alphas_t = get_by_idx(torch.sqrt(1.0 / self.alphas), t, xt.shape)
        predicted_noise = self.model(xt, t)
        return (sqrt_recip_alphas_t * (xt - sqrt_one_minus_bar_alphas_t * predicted_noise)
                + sqrt_one_minus_bar_alphas_prev_t * predicted_noise)

    def sample(self, batch_size, x=None, get_xt=False):
        '''Runs the reverse process from x = y_T (N(0, I) if None); returns the list of states, z = list[-1].'''
        if x is None:
            x = torch.randn(batch_size, self.latent_size, device=self.device)
        xs = [x] if get_xt else []
        for t in reversed(range(self.timesteps)):
            x = self.p_sample(x, torch.full((batch_size,), t, device=self.device, dtype=torch.long))
            xs.append(x)
        return xs

    def loss_at_step_t(self, x0, t):
        '''Huber noise-prediction loss at steps t for clean samples x0.'''
        noise = torch.randn_like(x0)
        return F.smooth_l1_loss(noise, self.model(self.q_sample(x0, t, noise), t))


def new_denoiser(latent_size, timesteps, device):
    return GaussianDiffusionNoise(ConditionalModel(latent_size, timesteps), timesteps, latent_size, device=device).to(device)


def train_prior_denoiser(sample_prior, latent_size, timesteps, device, iters=100000, batch_size=128, lr=1e-3):
    '''Warm-up of an x-independent denoiser on prior samples (sample_prior(n) -> n x latent_size).'''
    G = new_denoiser(latent_size, timesteps, device)
    opt = optim.Adam(G.parameters(), lr=lr)
    for i in range(iters):
        G.train()
        opt.zero_grad()
        t = torch.randint(0, timesteps, (batch_size,), device=device).long()
        loss = G.loss_at_step_t(sample_prior(batch_size), t)
        loss.backward()
        opt.step()
        if i % 10000 == 0:
            log.info(f'Prior denoiser warm-up step {i}: loss {loss.item():.4f}')
    return G
