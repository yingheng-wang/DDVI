'''DDVI with an x-conditional reverse process q(z|y, x), trained by wake-sleep (Tables 1, 7, 9; second training stage of
Appendix H). The latent is standardized with the prior's mean / std (u = (z - m) / s).
- Forward process r on u: Gaussian diffusion with a linear beta schedule (beta_T / beta_1 = 200) scaled so that
  alpha_bar_T = ab_T.
- q(y_T|x) = N(mu(x), diag exp(log_var(x))); reverse steps use the noise predictor eps(y_t, t, h(x)) with the forward
  posterior variance (beta_1 for the last step); the predicted u_0 is clipped smoothly to [-clip_u, clip_u].
- Wake: per-pixel reconstruction - kld_weight * regularizer along a sampled stochastic trajectory, where the regularizer
  is the closed-form surrogate log p(z) - KL(q(y_T|x) || N(0, I)) - sum_t KL(q(y_{t-1}|y_t, x) || r(y_{t-1}|y_t, u)).
- Sleep (sleep_ct steps per wake step, separate optimizer over the inference networks): z ~ p(z), a fantasy
  x_hat ~ p(x|z) (Bernoulli for MNIST, Gaussian with a running-average ML variance for CIFAR-10), noise-prediction loss
  of the x_hat-conditional denoiser (context dropped with probability uncond_prob)
  + yT_weight * KL(r(y_T|u) || q(y_T|x_hat)).
- Initialization: encoder trunk and decoder from the simplified DDVI (init_from), denoiser pre-trained on prior samples
  with a null context (pretrain_iters), decoder fixed for the first freeze_P_epochs epochs.
- Evaluation: deterministic DDIM reverse process z = F_x(y_T); the exact ELBO uses
  log q(z|x) = log q(y_T|x) - log |det dF_x/dy_T|.'''
import logging
import math

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ..nets import CondDenoiser, EvalAwareMLP, encoder_decoder
from ..utils import flow_logdet, kl_to_standard_normal, normal_logpdf
from .base import BaseModel

log = logging.getLogger(__name__)


def linear_betas_to_target(T, ab_T, ratio=200.0):
    '''Linear beta schedule (beta_T / beta_1 = ratio) scaled so that prod(1 - beta_t) = ab_T.'''
    lo, hi = 1e-8, 1.0
    shape = np.linspace(1.0, ratio, T)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if np.prod(1 - np.clip(mid * shape, 0, 0.999)) > ab_T:
            lo = mid
        else:
            hi = mid
    return np.clip(0.5 * (lo + hi) * shape, 0, 0.999)


class DDVI(BaseModel):
    ckpt_modules = ('Q', 'P', 'D')

    def build(self):
        cfg, m = self.cfg, self.cfg.model
        L = self.latent_size
        self.T, self.ctx_dim = m.timesteps, m.context_size
        self.clip_u = m.clip_u
        self.sigma2 = None   # ML variance of the Gaussian decoder (CIFAR-10 fantasies)
        self.Q, self.P = encoder_decoder(cfg, [L, L, self.ctx_dim], mlp=EvalAwareMLP, dropout=m.dropout)
        self.D = CondDenoiser(L, self.ctx_dim, hidden=m.den_hidden, n_blocks=m.den_blocks)
        with torch.no_grad():
            ps = self.prior.sample(20000).float()
        self.z_mean, self.z_std = ps.mean(0).to(self.device), ps.std(0).to(self.device)

        betas = torch.tensor(linear_betas_to_target(self.T, m.ab_T), dtype=torch.float32)
        alphas = 1 - betas
        ab = torch.cumprod(alphas, 0)
        ab_prev = torch.cat([torch.ones(1), ab[:-1]])
        post_var = betas * (1 - ab_prev) / (1 - ab)
        post_var[0] = betas[0]   # variance of the last reverse step q(u|y_1)
        self.sched = {k: v.to(self.device) for k, v in dict(
            ab=ab, post_var=post_var,
            coef_u=torch.sqrt(ab_prev) * betas / (1 - ab),
            coef_y=torch.sqrt(alphas) * (1 - ab_prev) / (1 - ab)).items()}

        if self.training and m.init_from:
            self._warm_start(m.init_from)
        for net in (self.Q, self.P, self.D):
            net.to(self.device)
        self.optim = optim.Adam(list(self.P.parameters()) + list(self.Q.parameters()) + list(self.D.parameters()), lr=self.lr)
        self.sleep_optim = optim.Adam(list(self.Q.parameters()) + list(self.D.parameters()), lr=m.sleep_lr)

    def _warm_start(self, path):
        '''Encoder trunk and decoder from a simplified-DDVI checkpoint (the y_T heads are not copied).'''
        ck = torch.load(path, map_location='cpu')
        q_new = self.Q.state_dict()
        copied = [k for k, v in ck['Q'].items() if 'lin_outs.' not in k and k in q_new and q_new[k].shape == v.shape]
        q_new.update({k: ck['Q'][k] for k in copied})
        self.Q.load_state_dict(q_new)
        self.P.load_state_dict(ck['P'])
        log.info(f'Warm start from {path}: decoder + encoder {copied}')

    # ----- reverse process -----
    def _to_z(self, u):
        return self.z_mean + self.z_std * u

    def _clip(self, u0):
        '''Identity on [-0.8c, 0.8c], then a tanh saturation to +-c with continuous slope (keeps the map invertible).'''
        c = self.clip_u
        a = 0.8 * c
        au = u0.abs()
        return torch.where(au <= a, u0, torch.sign(u0) * (a + (c - a) * torch.tanh((au - a) / (c - a))))

    def _step(self, y, t, ctx):
        '''Noise prediction and clipped u_0 prediction at step t.'''
        s = self.sched
        eps = self.D(y, torch.full((y.size(0),), t, device=self.device, dtype=torch.long), ctx)
        u0 = self._clip((y - torch.sqrt(1 - s['ab'][t]) * eps) / torch.sqrt(s['ab'][t]))
        return eps, u0

    def _ddim_map(self, y, ctx):
        '''Deterministic reverse process u = F_x(y_T) (DDIM update).'''
        s = self.sched
        for t in reversed(range(self.T)):
            eps, u0 = self._step(y, t, ctx)
            y = torch.sqrt(s['ab'][t - 1]) * u0 + torch.sqrt(1 - s['ab'][t - 1]) * eps if t > 0 else u0
        return y

    def _stochastic_chain(self, y, ctx):
        '''Stochastic reverse process from y = y_T; returns u = y_0 and the steps (t, y_t, mean of q(y_{t-1}|y_t, x)).'''
        s = self.sched
        steps = []
        for t in reversed(range(self.T)):
            eps, u0 = self._step(y, t, ctx)
            mean = s['coef_u'][t] * u0 + s['coef_y'][t] * y
            steps.append((t, y, mean))
            y = mean + torch.sqrt(s['post_var'][t]) * torch.randn_like(y)
        return y, steps

    # ----- training -----
    def _wake_loss(self, x):
        s = self.sched
        mu, log_var, ctx = self.Q(x)
        u, steps = self._stochastic_chain(mu + torch.exp(0.5 * log_var) * torch.randn_like(mu), ctx)
        z = self._to_z(u)
        rec = self.criterion_nored(self.P(z), x).view(x.size(0), -1).mean(-1)
        reg = -kl_to_standard_normal(mu, log_var)
        for t, y_t, mean_q in steps:
            if t > 0:
                mean_r = s['coef_u'][t] * u + s['coef_y'][t] * y_t
                reg = reg - torch.sum((mean_q - mean_r) ** 2, -1) / (2 * s['post_var'][t])
        reg = reg + self.prior.evaluate(z, nored=True) + torch.log(self.z_std).sum()
        return (rec - self.kld_weights[self.ep] * reg).mean(), rec.mean(), reg.mean()

    def _sleep_loss(self, batch_size):
        m, s = self.cfg.model, self.sched
        z = self.prior.sample(batch_size).float().to(self.device)
        with torch.no_grad():
            self.P.eval()
            x_hat = self.P(z)
            self.P.train()
            if self.mnist:
                x_hat = torch.bernoulli(x_hat)
            elif self.sigma2 is not None:
                x_hat = x_hat + math.sqrt(self.sigma2) * torch.randn_like(x_hat)
        mu, log_var, ctx = self.Q(x_hat)
        u = (z - self.z_mean) / self.z_std
        t = torch.randint(0, self.T, (batch_size,), device=self.device)
        eps = torch.randn_like(u)
        ab_t = s['ab'][t][:, None]
        y_t = torch.sqrt(ab_t) * u + torch.sqrt(1 - ab_t) * eps
        if m.uncond_prob > 0:
            ctx = ctx * (torch.rand(batch_size, 1, device=self.device) >= m.uncond_prob).float()
        loss = F.mse_loss(self.D(y_t, t, ctx), eps)
        ab_T = s['ab'][-1]
        kl_T = 0.5 * torch.sum(log_var - torch.log(1 - ab_T)
                               + ((1 - ab_T) + (torch.sqrt(ab_T) * u - mu) ** 2) / log_var.exp() - 1, -1)
        return loss + m.yT_weight * kl_T.mean()

    def _pretrain_denoiser(self, iters, batch_size=512, lr=1e-3):
        '''Warm-up of the reverse process on prior samples with a null x-context.'''
        opt = optim.Adam(self.D.parameters(), lr=lr)
        s = self.sched
        for i in range(iters):
            u = (self.prior.sample(batch_size).float().to(self.device) - self.z_mean) / self.z_std
            t = torch.randint(0, self.T, (batch_size,), device=self.device)
            eps = torch.randn_like(u)
            ab_t = s['ab'][t][:, None]
            y_t = torch.sqrt(ab_t) * u + torch.sqrt(1 - ab_t) * eps
            loss = F.mse_loss(self.D(y_t, t, torch.zeros(batch_size, self.ctx_dim, device=self.device)), eps)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if i % 5000 == 0:
                log.info(f'Denoiser pre-training step {i}: loss {loss.item():.4f}')

    def _train_epoch(self, dataloader):
        m = self.cfg.model
        if self.ep == 0 and m.pretrain_iters:
            self._pretrain_denoiser(m.pretrain_iters)
        all_res = 0
        for inputs in dataloader:
            inputs = inputs.to(self.device)
            self.P.train()
            self.Q.train()
            self.D.train()
            self.optim.zero_grad(set_to_none=True)
            loss, rec, reg = self._wake_loss(inputs)
            if not self.mnist:   # running ML variance of the Gaussian decoder (rec = mean per-pixel squared error)
                r = float(rec.detach())
                self.sigma2 = r if self.sigma2 is None else 0.99 * self.sigma2 + 0.01 * r
            loss.backward()
            if self.ep < m.freeze_P_epochs:
                for p in self.P.parameters():
                    p.grad = None
            self.optim.step()
            sleep_loss = torch.zeros(())
            for _ in range(m.sleep_ct):
                self.sleep_optim.zero_grad(set_to_none=True)
                sleep_loss = self._sleep_loss(self.batch_size)
                sleep_loss.backward()
                self.sleep_optim.step()
            all_res += np.asarray([loss.item(), rec.item(), reg.item(), sleep_loss.item()])
        return all_res / len(dataloader)

    # ----- evaluation -----
    def _encode(self, x):
        mu, log_var, ctx = self.Q(x)
        return self._to_z(self._ddim_map(mu + torch.exp(0.5 * log_var) * torch.randn_like(mu), ctx))

    def elbo_terms(self, x, need_q=True):
        mu, log_var, ctx = self.Q(x)
        y = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)
        flow = lambda yy: self._to_z(self._ddim_map(yy, ctx))
        if not need_q:
            return self.rec_sum(self.P(flow(y)), x), None, None, None
        z, ld = flow_logdet(flow, y)
        r = self.rec_sum(self.P(z), x)
        return r, self.prior.evaluate(z, nored=True), normal_logpdf(y, mu, log_var) - ld, torch.zeros_like(r)

    def _extra_state(self):
        return {'z_mean': self.z_mean, 'z_std': self.z_std, 'sigma2': self.sigma2}

    def _load_extra_state(self, ck):
        if 'z_mean' in ck:   # checkpoints of the original code base do not store the standardization
            self.z_mean, self.z_std = ck['z_mean'].to(self.device), ck['z_std'].to(self.device)
            self.sigma2 = ck.get('sigma2')
