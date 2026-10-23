'''Baselines with Gaussian / IAF / hierarchical posteriors (AEVB, AEVB-IAF, H-AEVB) and their semi-supervised versions.
Training loss: per-pixel reconstruction + w_t * (E[log q(z|x)] - p_weight * E[log p(z)]) with the KDE prior density,
plus w_t * cls_weight * cross-entropy on the labeled examples in the semi-supervised versions.'''
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from ..flows import IAF
from ..nets import encoder_decoder, get_q2
from ..priors import GaussPrior
from ..utils import cs_entropy, label_kl, sample
from .base import BaseModel


class AEVB(BaseModel):
    def build(self):
        self.p_weight = self.cfg.model.p_weight
        self.Q, self.P = encoder_decoder(self.cfg, self._out_sizes())
        self.Q.to(self.device)
        self.P.to(self.device)
        self.optim = optim.Adam(list(self.P.parameters()) + list(self.Q.parameters()), lr=self.lr)

    def _out_sizes(self):
        return [self.latent_size, self.latent_size]

    def _posterior(self, outs):
        '''z ~ q(z|x) and log q(z|x) from the encoder outputs.'''
        mu, var_s = outs[0], outs[1]
        return sample(mu, torch.log(torch.sigmoid(var_s)), ret_log_prob=True)

    def _train_mode(self):
        self.P.train()
        self.Q.train()

    def _train_epoch(self, dataloader):
        all_res = 0
        for inputs in dataloader:
            inputs = inputs.to(self.device)
            self._train_mode()
            self.optim.zero_grad()
            rep, log_q = self._posterior(self.Q(inputs))
            reg_loss1 = torch.mean(log_q)
            rec_loss = self.criterion(self.P(rep), inputs)
            reg_loss2 = self.prior.evaluate(rep)
            loss = rec_loss + self.kld_weights[self.ep] * (reg_loss1 - self.p_weight * reg_loss2)
            loss.backward()
            self.optim.step()
            all_res += np.asarray([rec_loss.item(), reg_loss1.item(), reg_loss2.item()])
        return all_res / len(dataloader)

    def _encode(self, x):
        return self._posterior(self.Q(x))[0]

    def elbo_terms(self, x, need_q=True):
        z, log_q = self._posterior(self.Q(x))
        r = self.rec_sum(self.P(z), x)
        if not need_q:
            return r, None, None, None
        return r, self.prior.evaluate(z, nored=True), log_q, torch.zeros_like(r)


class AEVB_IAF(AEVB):
    '''Gaussian base distribution followed by 4 IAF layers conditioned on a 10-d context from the encoder.'''
    ckpt_modules = ('Q', 'P', 'iaf_layers')

    def build(self):
        self.p_weight = self.cfg.model.p_weight
        self.Q, self.P = encoder_decoder(self.cfg, self._out_sizes())
        self.iaf_layers = torch.nn.ModuleList([IAF(self.latent_size, parity=i % 2) for i in range(4)])
        for net in (self.Q, self.P, self.iaf_layers):
            net.to(self.device)
        self.optim = optim.Adam(list(self.P.parameters()) + list(self.Q.parameters()) + list(self.iaf_layers.parameters()),
                                lr=self.lr)

    def _out_sizes(self):
        return [self.latent_size, self.latent_size, self.cfg.model.context_size]

    def _posterior(self, outs):
        mu, var_s, context = outs[0], outs[1], outs[2]
        rep, log_q = sample(mu, torch.log(torch.sigmoid(var_s)), ret_log_prob=True)
        for layer in self.iaf_layers:
            rep, ld = layer(rep, context)
            log_q = log_q - ld
        return rep, log_q


class H_AEVB(AEVB):
    '''Hierarchical posterior q(z1|x) q(z2|x, z1) (both IAF), with p(z1) = N(0, I) and z2 decoded under the prior p.
    The ELBO is the auxiliary-variable bound with reverse model N(0, I) for z1.'''
    ckpt_modules = ('Q', 'P', 'iaf_layers', 'Q2', 'iaf_layers_2')

    def build(self):
        self.p_weight = self.cfg.model.p_weight
        self.gauss_prior = GaussPrior(self.latent_size)
        self.Q, self.P = encoder_decoder(self.cfg, [self.latent_size, self.latent_size, self.cfg.model.context_size])
        self.Q2 = get_q2(self.cfg)
        self.iaf_layers = torch.nn.ModuleList([IAF(self.latent_size, parity=i % 2, nh=64) for i in range(4)])
        self.iaf_layers_2 = torch.nn.ModuleList([IAF(self.latent_size, parity=i % 2, nh=64) for i in range(4)])
        for net in (self.Q, self.P, self.iaf_layers, self.Q2, self.iaf_layers_2):
            net.to(self.device)
        self.optim = optim.Adam(list(self.P.parameters()) + list(self.Q.parameters()) + list(self.iaf_layers.parameters())
                                + list(self.Q2.parameters()) + list(self.iaf_layers_2.parameters()), lr=self.lr)

    def _two_level(self, x):
        '''z1, log q(z1|x), z2, log q(z2|x, z1).'''
        mu, var_s, context = self.Q(x)
        z1, lq1 = sample(mu, torch.log(torch.sigmoid(var_s)), ret_log_prob=True)
        for layer in self.iaf_layers:
            z1, ld = layer(z1, context)
            lq1 = lq1 - ld
        mu, var_s, context = self.Q2(x, z1)
        z2, lq2 = sample(mu, torch.log(torch.clip(torch.sigmoid(var_s), 1e-6, 1)), ret_log_prob=True)
        for layer in self.iaf_layers_2:
            z2, ld = layer(z2, context)
            lq2 = lq2 - ld
        return z1, lq1, z2, lq2

    def _train_mode(self):
        self.P.train()
        self.Q.train()
        self.Q2.train()

    def _train_epoch(self, dataloader):
        all_res = 0
        for inputs in dataloader:
            inputs = inputs.to(self.device)
            self._train_mode()
            self.optim.zero_grad()
            z1, lq1, z2, lq2 = self._two_level(inputs)
            rec_loss = self.criterion(self.P(z2), inputs)
            reg1 = torch.mean(lq1) - self.p_weight * self.gauss_prior.evaluate(z1)
            reg2 = torch.mean(lq2) - self.p_weight * self.prior.evaluate(z2)
            loss = rec_loss + self.kld_weights[self.ep] * (reg1 + reg2)
            loss.backward()
            self.optim.step()
            all_res += np.asarray([rec_loss.item(), reg1.item(), reg2.item()])
        return all_res / len(dataloader)

    def _encode(self, x):
        return self._two_level(x)[2]

    def elbo_terms(self, x, need_q=True):
        z1, lq1, z2, lq2 = self._two_level(x)
        r = self.rec_sum(self.P(z2), x)
        if not need_q:
            return r, None, None, None
        log_p = self.gauss_prior.evaluate(z1, nored=True) + self.prior.evaluate(z2, nored=True)
        return r, log_p, lq1 + lq2, torch.zeros_like(r)


class SemiMixin:
    '''Semi-supervised objective: the encoder has an extra class head q(l|x); labeled examples use log p(z|l),
    unlabeled ones E_q(l|x)[log p(z|l)] plus 0.001 * the entropy of q(l|x), and the labeled examples add a
    cross-entropy term weighted by cls_weight * w_t.'''

    def _out_sizes(self):
        return super()._out_sizes() + [self.n_classes]

    def _train_epoch(self, dataloader):
        all_res = 0
        cls_weight = self.cfg.model.cls_weight
        for inputs, targets in dataloader:
            inputs = inputs.to(self.device)
            targets_cpu = targets
            targets = targets.to(self.device)
            self._train_mode()
            self.optim.zero_grad()
            outs = self.Q(inputs)
            cs_logits = outs[-1]
            cs = F.softmax(cs_logits, 1)
            rep, log_q = self._posterior(outs)
            reg_loss1 = torch.mean(log_q)
            rec_loss = self.criterion(self.P(rep), inputs)

            unl = torch.arange(inputs.size(0))[targets_cpu == self.n_classes]
            unlabeled_loss = torch.zeros(1, device=self.device)
            labeled_loss = torch.zeros(1, device=self.device)
            for i in range(self.n_classes):
                unlabeled_loss = unlabeled_loss + torch.sum(cs[unl, i] * self.prior.evaluate_partition(rep[unl], i))
                labeled_loss = labeled_loss + torch.sum(self.prior.evaluate_partition(rep[targets_cpu == i], i))
            reg_loss2 = (unlabeled_loss / len(unl) + labeled_loss / max(1, inputs.size(0) - len(unl))
                         + 0.001 * cs_entropy(cs[unl]))

            lab = targets_cpu != self.n_classes
            cls_loss = (F.cross_entropy(cs_logits[lab], targets[lab]) if lab.any()
                        else torch.zeros(1, device=self.device))
            w = self.kld_weights[self.ep]
            loss = rec_loss + w * (reg_loss1 - self.p_weight * reg_loss2) + cls_weight * w * cls_loss
            loss.backward()
            self.optim.step()
            all_res += np.asarray([rec_loss.item(), reg_loss1.item(), reg_loss2.item(), cls_loss.item()])
        return all_res / len(dataloader)

    def elbo_terms(self, x, need_q=True):
        outs = self.Q(x)
        z, log_q = self._posterior(outs)
        r = self.rec_sum(self.P(z), x)
        if not need_q:
            return r, None, None, None
        q_l, lkl = label_kl(outs[-1])
        log_p = sum(q_l[:, c] * self.prior.evaluate_partition(z, c) for c in range(self.n_classes))
        return r, log_p, log_q, lkl


class AEVB_semi(SemiMixin, AEVB):
    pass


class AEVB_IAF_semi(SemiMixin, AEVB_IAF):
    pass
