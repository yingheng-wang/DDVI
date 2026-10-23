'''Encoders, decoders and small denoising networks shared by all methods.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiWayLinear(nn.Module):
    '''MLP with LeakyReLU(0.2) hidden layers and one linear output head per entry of out_sizes.
    Dropout is applied in training and evaluation alike (the setting of all reported baseline results).'''
    def __init__(self, in_size, hidden_sizes, out_sizes, sigmoid=False, bn=False, dropout=None):
        super().__init__()
        self.sigmoid = sigmoid
        self.use_bn = bn
        self.dropout = dropout
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        if isinstance(out_sizes, int):
            out_sizes = [out_sizes]
        self.lins = nn.ModuleList([nn.Linear(in_size, hidden_sizes[0])] +
                                  [nn.Linear(hidden_sizes[i], hidden_sizes[i + 1]) for i in range(len(hidden_sizes) - 1)])
        self.lin_outs = nn.ModuleList([nn.Linear(hidden_sizes[-1], out_size) for out_size in out_sizes])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_size) for hidden_size in hidden_sizes])
        self.leaky_relu = nn.LeakyReLU(0.2)

    def _dropout(self, x):
        return F.dropout(x, self.dropout)

    def forward(self, x):
        for i, lin in enumerate(self.lins):
            x = lin(x)
            if self.use_bn:
                x = self.bns[i](x)
            x = self.leaky_relu(x)
            if self.dropout is not None:
                x = self._dropout(x)
        res = [lin_out(x) for lin_out in self.lin_outs]
        if self.sigmoid:
            res = [torch.sigmoid(out) for out in res]
        return res if len(self.lin_outs) > 1 else res[0]


class EvalAwareMLP(MultiWayLinear):
    '''MultiWayLinear whose dropout is active only in training mode (used by DDVI with the x-conditional reverse process).'''
    def _dropout(self, x):
        return F.dropout(x, self.dropout, training=self.training)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def _conv_trunk():
    return nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=4, stride=2), nn.BatchNorm2d(32), nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=5, stride=2), nn.BatchNorm2d(64), nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=5, stride=2), nn.BatchNorm2d(128), nn.ReLU(),
        Flatten())


class ConvNet(nn.Module):
    '''CIFAR-10 encoder: convolutional trunk (128 features) followed by the head network fc.'''
    def __init__(self, fc):
        super().__init__()
        self.convs = _conv_trunk()
        self.fc = fc

    def forward(self, x):
        return self.fc(self.convs(x))


class BareConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.convs = _conv_trunk()

    def forward(self, x):
        return self.convs(x)


class TransposedConvNet(nn.Module):
    '''CIFAR-10 decoder; the tanh output matches the range [-1, 1] of the normalized images.'''
    def __init__(self, input_size=2):
        super().__init__()
        self.feature_dim = 64
        self.fc = nn.Linear(input_size, self.feature_dim)
        self.t_convs = nn.Sequential(
            nn.ConvTranspose2d(self.feature_dim, 64, kernel_size=5, stride=2), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2), nn.BatchNorm2d(16), nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=4, stride=1), nn.Tanh())

    def forward(self, z):
        x = F.relu(self.fc(z))
        return self.t_convs(x.view(-1, self.feature_dim, 1, 1))


def encoder_decoder(cfg, out_sizes, mlp=MultiWayLinear, dropout=0.2):
    '''Encoder Q (one output head per entry of out_sizes) and decoder P for the dataset of cfg.'''
    H, L = cfg.model.hidden_size, cfg.model.latent_size
    if cfg.dataset.name.startswith('mnist'):
        Q = mlp(784, [H, H], out_sizes, bn=cfg.model.bn, dropout=dropout)
        P = mlp(L, [H, H], 784, sigmoid=True, dropout=dropout)
    elif cfg.dataset.name.startswith('cifar10'):
        Q = ConvNet(mlp(128, [H], out_sizes, bn=cfg.model.bn, dropout=dropout))
        P = TransposedConvNet(L)
    else:
        raise ValueError(cfg.dataset.name)
    return Q, P


class Q2(nn.Module):
    '''Second-level encoder of H-AEVB: q(z2 | x, z1).'''
    def __init__(self, img_processor, fc):
        super().__init__()
        self.img_processor = img_processor
        self.fc = fc

    def forward(self, x, z):
        return self.fc(torch.cat((self.img_processor(x), z), 1))


def get_q2(cfg):
    H, L = cfg.model.hidden_size, cfg.model.latent_size
    if cfg.dataset.name.startswith('mnist'):
        img_processor = MultiWayLinear(784, [H, H], 128, bn=cfg.model.bn, dropout=0.2)
    else:
        img_processor = BareConvNet()
    fc = MultiWayLinear(128 + L, [24, 24], [L, L, cfg.model.context_size], bn=cfg.model.bn, dropout=0.2)
    return Q2(img_processor, fc)


# ----- denoisers -----
class ConditionalLinear(nn.Module):
    def __init__(self, num_in, num_out, n_steps):
        super().__init__()
        self.num_out = num_out
        self.lin = nn.Linear(num_in, num_out)
        self.embed = nn.Embedding(n_steps, num_out)
        self.embed.weight.data.uniform_()

    def forward(self, x, t):
        return self.embed(t).view(-1, self.num_out) + self.lin(x)


class ConditionalModel(nn.Module):
    '''x-independent noise predictor eps(y_t, t) of the simplified DDVI.'''
    def __init__(self, z_size, n_steps):
        super().__init__()
        self.lin1 = ConditionalLinear(z_size, 128, n_steps)
        self.lin2 = ConditionalLinear(128, 128, n_steps)
        self.lin3 = ConditionalLinear(128, 128, n_steps)
        self.lin4 = ConditionalLinear(128, 128, n_steps)
        self.lin_mean = nn.Linear(128, z_size)

    def forward(self, z, t):
        z = F.relu(self.lin1(z, t))
        z = F.relu(self.lin2(z, t))
        z = F.relu(self.lin3(z, t))
        z = F.relu(self.lin4(z, t))
        return self.lin_mean(z)


class ConditionalModelWAUX(nn.Module):
    '''Drift network u(z, t, h(x)) of PIS.'''
    def __init__(self, in_size, n_steps, out_size):
        super().__init__()
        self.lin1 = ConditionalLinear(in_size, 128, n_steps)
        self.lin2 = ConditionalLinear(128, 128, n_steps)
        self.lin3 = ConditionalLinear(128, 128, n_steps)
        self.lin4 = ConditionalLinear(128, 128, n_steps)
        self.lin_mean = nn.Linear(128, out_size)

    def forward(self, z, a, t):
        za = F.relu(self.lin1(torch.cat((z, a), 1), t))
        za2 = F.relu(self.lin2(za, t))
        za2 = F.relu(self.lin3(za2, t)) + za
        za2 = F.relu(self.lin4(za2, t))
        return self.lin_mean(za2)


def timestep_embedding(t, dim):
    half = dim // 2
    freqs = torch.exp(-math.log(10000.) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], -1)


class FiLMBlock(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.lin1 = nn.Linear(hidden, hidden)
        self.film = nn.Linear(hidden, 2 * hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        nn.init.zeros_(self.lin2.weight)
        nn.init.zeros_(self.lin2.bias)

    def forward(self, h, cond):
        x = F.silu(self.lin1(self.norm(h)))
        scale, shift = self.film(cond).chunk(2, -1)
        x = x * (1 + scale) + shift
        return h + self.lin2(F.silu(x))


class CondDenoiser(nn.Module):
    '''x-conditional noise predictor eps(y_t, t, h(x)): FiLM conditioning on [time embedding, context] in every
    residual block; the output layer starts at zero.'''
    def __init__(self, z_dim, ctx_dim, hidden=256, n_blocks=4, temb_dim=64):
        super().__init__()
        self.temb_dim = temb_dim
        self.cond = nn.Sequential(nn.Linear(temb_dim + ctx_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.inp = nn.Linear(z_dim, hidden)
        self.blocks = nn.ModuleList([FiLMBlock(hidden) for _ in range(n_blocks)])
        self.out_norm = nn.LayerNorm(hidden)
        self.out = nn.Linear(hidden, z_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, y, t, ctx):
        cond = self.cond(torch.cat([timestep_embedding(t, self.temb_dim), ctx], -1))
        h = self.inp(y)
        for blk in self.blocks:
            h = blk(h, cond)
        return self.out(F.silu(self.out_norm(h)))
