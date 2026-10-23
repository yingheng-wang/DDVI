import math

import numpy as np
import torch
import torch.optim as optim

from ..nets import ConditionalModelWAUX, encoder_decoder
from .base import BaseModel


class PIS(BaseModel):
    '''Amortized Path Integral Sampler (Zhang & Chen, 2021) as the variational posterior q(z|x) (our reimplementation).
    z follows dz = u(z, t, h(x)) dt + sigma dW on t in [0, 1] from z_0 = 0 (Euler-Maruyama, `timesteps` steps); the
    reference process (u = 0) ends at N(0, sigma^2 I). By Girsanov's theorem,
        ELBO = E[log p(x|z_1) + log p(z_1) - log N(z_1; 0, sigma^2 I) - sum_k ||u_k||^2 dt / (2 sigma^2)],
    optimized on-policy by backpropagating through the sampled path.'''
    ckpt_modules = ('Q', 'P', 'U')

    def build(self):
        m = self.cfg.model
        self.p_weight, self.context_size, self.n_steps, self.sigma = m.p_weight, m.context_size, m.timesteps, m.sigma
        self.Q, self.P = encoder_decoder(self.cfg, self.context_size)
        self.U = ConditionalModelWAUX(self.latent_size + self.context_size, self.n_steps, self.latent_size)
        for net in (self.Q, self.P, self.U):
            net.to(self.device)
        self.optim = optim.Adam(list(self.P.parameters()) + list(self.Q.parameters()) + list(self.U.parameters()), lr=self.lr)

    def _sample_path(self, h):
        batch_size = h.size(0)
        dt = 1.0 / self.n_steps
        z = torch.zeros(batch_size, self.latent_size, device=self.device)
        path_kl = torch.zeros(batch_size, device=self.device)
        for k in range(self.n_steps):
            u = self.U(z, h, torch.full((batch_size,), k, device=self.device, dtype=torch.long))
            path_kl = path_kl + 0.5 * torch.sum(u ** 2, -1) * dt / self.sigma ** 2
            z = z + u * dt + self.sigma * math.sqrt(dt) * torch.randn_like(z)
        return z, path_kl

    def _log_ref(self, z):
        '''log N(z; 0, sigma^2 I), the terminal density of the uncontrolled reference process.'''
        return (-0.5 * torch.sum(z ** 2, -1) / self.sigma ** 2
                - self.latent_size * (math.log(self.sigma) + 0.5 * math.log(2 * math.pi)))

    def _train_epoch(self, dataloader):
        all_res = 0
        for inputs in dataloader:
            inputs = inputs.to(self.device)
            self.P.train()
            self.Q.train()
            self.U.train()
            self.optim.zero_grad()
            z, path_kl = self._sample_path(self.Q(inputs))
            rec_loss = self.criterion(self.P(z), inputs)
            reg_loss1 = torch.mean(path_kl + self._log_ref(z))   # path-space analogue of E[log q]
            reg_loss2 = self.prior.evaluate(z)
            loss = rec_loss + self.kld_weights[self.ep] * (reg_loss1 - self.p_weight * reg_loss2)
            loss.backward()
            self.optim.step()
            all_res += np.asarray([rec_loss.item(), reg_loss1.item(), reg_loss2.item()])
        return all_res / len(dataloader)

    def _encode(self, x):
        return self._sample_path(self.Q(x))[0]

    def elbo_terms(self, x, need_q=True):
        z, path_kl = self._sample_path(self.Q(x))
        r = self.rec_sum(self.P(z), x)
        if not need_q:
            return r, None, None, None
        return r, self.prior.evaluate(z, nored=True), path_kl + self._log_ref(z), torch.zeros_like(r)
