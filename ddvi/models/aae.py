'''Adversarial autoencoder baselines (AAEB), after https://github.com/fducau/AAE_pytorch. A discriminator on the latents
(and, semi-supervised, on [z, one-hot label with an extra "unlabeled" class]) matches the aggregate posterior to the
prior. AAEB has no ELBO.'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..nets import encoder_decoder
from .base import BaseModel

EPS = 1e-15


class GaussianNoise(nn.Module):
    '''Multiplicative Gaussian noise (relative standard deviation sigma) on the discriminator inputs, in training mode.'''
    def __init__(self, sigma=0.3):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training and self.sigma != 0:
            x = x + torch.randn_like(x) * (self.sigma * x.detach())
        return x


class D_net(nn.Module):
    def __init__(self, input_size, hidden_size=1000):
        super().__init__()
        self.hidden_layer_1 = nn.Linear(input_size, hidden_size)
        self.hidden_layer_2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.noise = GaussianNoise()

    def forward(self, x):
        x = F.dropout(self.leaky_relu(self.hidden_layer_1(self.noise(x))), 0.2)
        x = F.dropout(self.leaky_relu(self.hidden_layer_2(x)), 0.2)
        return torch.sigmoid(self.output_layer(x))


class AAEB(BaseModel):
    ckpt_modules = ('Q', 'P', 'D_gauss')

    def __init__(self, cfg, training=True):
        super().__init__(cfg, training)
        self.criterion = torch.nn.MSELoss()   # for MNIST as well

    def _disc_input_size(self):
        return self.latent_size

    def build(self):
        self.Q, self.P = encoder_decoder(self.cfg, self.latent_size)
        self.D_gauss = D_net(self._disc_input_size())
        for net in (self.Q, self.P, self.D_gauss):
            net.to(self.device)
        self.ae_optim = optim.Adam(list(self.P.parameters()) + list(self.Q.parameters()), lr=self.lr)
        self.disc_optim = optim.Adam(self.D_gauss.parameters(), lr=self.lr)
        self.gen_optim = optim.Adam(self.Q.parameters(), lr=self.lr)
        self.schedulers = [optim.lr_scheduler.MultiStepLR(o, milestones=[150], gamma=0.1)
                           for o in (self.ae_optim, self.disc_optim, self.gen_optim)]

    def _call_schedulers(self):
        for s in self.schedulers:
            s.step()
        self.D_gauss.noise.sigma = max(0, 0.1 * (((self.num_epochs - (self.ep + 1)) // (self.num_epochs // 4)) - 1))

    def _disc_inputs(self, zs, labels):
        return zs

    def _prior_inputs(self, batch_size):
        return self.prior.sample(batch_size).to(self.device)

    def _batches(self, dataloader):
        for inputs in dataloader:
            yield inputs.to(self.device), None

    def _train_epoch(self, dataloader):
        total = np.zeros(3)
        for inputs, labels in self._batches(dataloader):
            # reconstruction phase
            self.Q.train()
            self.P.train()
            self.ae_optim.zero_grad()
            recon_loss = 0.5 * self.criterion(self.P(self.Q(inputs)), inputs)
            recon_loss.backward()
            self.ae_optim.step()

            # regularization phase: discriminator, then generator (the encoder)
            real = self._prior_inputs(inputs.size(0))
            self.Q.eval()
            self.D_gauss.train()
            self.disc_optim.zero_grad()
            D_real = self.D_gauss(real)
            D_fake = self.D_gauss(self._disc_inputs(self.Q(inputs), labels))
            D_loss = -6 * torch.mean(torch.log(D_real + EPS) + torch.log(1 - D_fake + EPS))
            D_loss.backward()
            self.disc_optim.step()

            self.Q.train()
            self.D_gauss.eval()
            self.gen_optim.zero_grad()
            G_loss = -6 * torch.mean(torch.log(self.D_gauss(self._disc_inputs(self.Q(inputs), labels)) + EPS))
            G_loss.backward()
            self.gen_optim.step()
            total += np.asarray([recon_loss.item(), D_loss.item(), G_loss.item()])
        return total / len(dataloader)

    def _encode(self, x):
        return self.Q(x)


class AAEB_semi(AAEB):
    def _disc_input_size(self):
        return self.latent_size + self.n_classes + 1

    def _call_schedulers(self):
        for s in self.schedulers:
            s.step()
        self.D_gauss.noise.sigma = 0.1 * ((self.num_epochs - (self.ep + 1)) // (self.num_epochs // 4))

    def _disc_inputs(self, zs, labels):
        return torch.cat((zs, F.one_hot(labels, self.n_classes + 1)), 1)

    def _prior_inputs(self, batch_size):
        zs, labels = self.prior.sample(batch_size, with_classes=True)
        # half of the prior samples are presented without their label
        labels[np.random.choice(len(labels), int(0.5 * len(labels)), replace=False)] = self.n_classes
        return self._disc_inputs(zs.to(self.device), labels.to(self.device))

    def _batches(self, dataloader):
        for inputs, labels in dataloader:
            yield inputs.to(self.device), labels.to(self.device)
