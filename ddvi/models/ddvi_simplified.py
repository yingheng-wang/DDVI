'''Simplified DDVI (Section 3.4, Appendix H): the reverse process does not depend on x. y_T ~ q(y_T|x) and
z = G(y_T) by the deterministic DDIM update of a denoiser G trained on prior samples, so the regularizer reduces to
KL(q(y_T|x) || N(0, I)) plus the denoising loss on p(z).
- DDVISimplified: unsupervised; first training stage of DDVI (initializes the encoder trunk and decoder of models/ddvi.py).
  G is warmed up on p(z) and then trained jointly with the encoder and decoder.
- DDVISemi: semi-supervised DDVI (Tables 2, 8). One denoiser G_l per partition, warmed up on p(z|l) and then kept fixed;
  q(z|x) = sum_l q(l|x) q(z|x, l).'''
import logging
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..diffusion import new_denoiser, train_prior_denoiser
from ..nets import encoder_decoder
from ..utils import cs_entropy, flow_logdet, kl_to_standard_normal, label_kl, normal_logpdf, sample
from .base import BaseModel

log = logging.getLogger(__name__)


class DDVISimplified(BaseModel):
    ckpt_modules = ('Q', 'G', 'P')

    def build(self):
        m = self.cfg.model
        self.timesteps, self.diff_weight = m.timesteps, m.diff_weight
        self.Q, self.P = encoder_decoder(self.cfg, [self.latent_size, self.latent_size])
        self.G = self._prior_denoiser('diff_model.pt', self.prior.sample)
        self.G.eval()
        for net in (self.Q, self.P, self.G):
            net.to(self.device)
        self.optim = optim.Adam(list(self.P.parameters()) + list(self.Q.parameters()) + list(self.G.parameters()), lr=self.lr)

    def _prior_denoiser(self, fname, sample_prior):
        '''Load fname from model.denoiser_dir if set (denoisers saved by an earlier run with the same prior and T),
        otherwise warm up a new denoiser on sample_prior and save it to save_folder/fname.'''
        if not self.training:   # the weights come from model.pt
            return new_denoiser(self.latent_size, self.timesteps, self.device)
        d = self.cfg.model.get('denoiser_dir', None)
        if d:
            G = new_denoiser(self.latent_size, self.timesteps, self.device)
            G.load_state_dict(torch.load(os.path.join(d, fname), map_location=self.device)['Q'])
            return G
        log.info(f'Warming up the prior denoiser ({fname})')
        G = train_prior_denoiser(sample_prior, self.latent_size, self.timesteps, self.device,
                                 iters=self.cfg.model.warmup_iters)
        os.makedirs(self.save_folder, exist_ok=True)
        torch.save({'Q': G.state_dict()}, os.path.join(self.save_folder, fname))
        return G

    def _train_epoch(self, dataloader):
        all_res = 0
        for inputs in dataloader:
            inputs = inputs.to(self.device)
            self.P.train()
            self.Q.train()
            self.G.train()
            prior_samples = self.prior.sample(self.batch_size)
            self.optim.zero_grad()
            mu, log_var = self.Q(inputs)
            z = self.G.sample(inputs.size(0), x=sample(mu, log_var))[-1]
            rec_loss = self.criterion(self.P(z), inputs)
            reg_loss = torch.mean(kl_to_standard_normal(mu, log_var))
            t = torch.randint(0, self.timesteps, (self.batch_size,), device=self.device).long()
            diff_loss = self.G.loss_at_step_t(prior_samples, t)
            loss = rec_loss + self.kld_weights[self.ep] * reg_loss + self.diff_weight * diff_loss
            loss.backward()
            self.optim.step()
            all_res += np.asarray([rec_loss.item(), reg_loss.item(), diff_loss.item()])
        return all_res / len(dataloader)

    def _encode(self, x):
        mu, log_var = self.Q(x)
        return self.G.sample(x.size(0), x=sample(mu, log_var))[-1]

    def elbo_terms(self, x, need_q=True):
        '''Exact ELBO of the deterministic map z = G(y_T): log q(z|x) = log q(y_T|x) - log |det dG/dy_T|.'''
        mu, log_var = self.Q(x)
        y = sample(mu, log_var)
        flow = lambda yy: self.G.sample(yy.size(0), x=yy)[-1]
        if not need_q:
            return self.rec_sum(self.P(flow(y)), x), None, None, None
        z, ld = flow_logdet(flow, y)
        r = self.rec_sum(self.P(z), x)
        return r, self.prior.evaluate(z, nored=True), normal_logpdf(y, mu, log_var) - ld, torch.zeros_like(r)


class DDVISemi(DDVISimplified):
    '''Training loss: reconstruction (summed over partitions with weights q(l|x) for unlabeled and the true label for
    labeled examples; per-pixel mean over the batch) + w * (KL(q(y_T|x) || N(0, I)) + 0.001 * entropy of q(l|x) on
    unlabeled examples) + w * cls_weight * cross-entropy on labeled examples. The latents of all partitions are decoded
    in one decoder call, so that BatchNorm sees a mixed batch, as at evaluation.'''

    def build(self):
        m = self.cfg.model
        self.timesteps = m.timesteps
        self.Q, self.P = encoder_decoder(self.cfg, [self.latent_size, self.latent_size, self.n_classes])
        self.G = nn.ModuleList([
            self._prior_denoiser(f'diff_model{i}.pt',
                                 lambda n, i=i: self.prior.sample(n, g_classes=np.full(n, i)))
            for i in range(self.n_classes)])
        self.G.eval()
        for net in (self.Q, self.P, self.G):
            net.to(self.device)
        self.optim = optim.Adam(list(self.P.parameters()) + list(self.Q.parameters()), lr=self.lr)

    def _train_epoch(self, dataloader):
        all_res = 0
        cls_weight = self.cfg.model.cls_weight
        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets_cpu = targets
            targets = targets.to(self.device)
            batch_size = inputs.size(0)
            self.P.train()
            self.Q.train()
            self.G.train()
            self.optim.zero_grad()
            mu, log_var, cs_logits = self.Q(inputs)
            cs = F.softmax(cs_logits, 1)
            y = sample(mu, log_var)
            unl = torch.arange(batch_size)[targets_cpu == self.n_classes]
            zs = [self.G[i].sample(batch_size, x=y)[-1] for i in range(self.n_classes)]
            recs = self.P(torch.cat(zs, 0)).split(batch_size, 0)
            bcast = (-1,) + (1,) * (inputs.dim() - 1)
            rec_loss = 0
            for i in range(self.n_classes):
                rec_loss = rec_loss + torch.sum(cs[unl, i].view(*bcast) * self.criterion_nored(recs[i][unl], inputs[unl]))
                rec_loss = rec_loss + torch.sum(self.criterion_nored(recs[i][targets_cpu == i], inputs[targets_cpu == i]))
            rec_loss = rec_loss / (batch_size * int(np.prod(inputs.size()[1:])))
            reg_loss = torch.mean(kl_to_standard_normal(mu, log_var)) + 0.001 * cs_entropy(cs[unl])
            lab = targets_cpu != self.n_classes
            cls_loss = (F.cross_entropy(cs_logits[lab], targets[lab]) if lab.any()
                        else torch.zeros(1, device=self.device))
            w = self.kld_weights[self.ep]
            loss = rec_loss + w * reg_loss + cls_weight * w * cls_loss
            loss.backward()
            self.optim.step()
            all_res += np.asarray([rec_loss.item(), reg_loss.item(), cls_loss.item()])
        return all_res / len(dataloader)

    def _encode(self, x):
        '''z from the denoiser of the predicted partition.'''
        mu, log_var, cs_logits = self.Q(x)
        pred = torch.argmax(cs_logits, 1)
        y = sample(mu, log_var)
        z = torch.zeros_like(y)
        for i in range(self.n_classes):
            if (pred == i).any():
                z[pred == i] = self.G[i].sample(int((pred == i).sum()), x=y[pred == i])[-1]
        return z

    def elbo_terms(self, x, need_q=True):
        mu, log_var, logits = self.Q(x)
        q_l, lkl = label_kl(logits)
        y = sample(mu, log_var)
        lqy = normal_logpdf(y, mu, log_var)
        r = log_p = log_q = 0
        for c in range(self.n_classes):
            flow = lambda yy: self.G[c].sample(yy.size(0), x=yy)[-1]
            if need_q:
                z, ld = flow_logdet(flow, y)
                log_p = log_p + q_l[:, c] * self.prior.evaluate_partition(z, c)
                log_q = log_q + q_l[:, c] * (lqy - ld)
            else:
                z = flow(y)
            r = r + q_l[:, c] * self.rec_sum(self.P(z), x)
        return (r, log_p, log_q, lkl) if need_q else (r, None, None, None)
