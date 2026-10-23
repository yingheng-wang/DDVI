'''Inverse autoregressive flow layers (AEVB-IAF, H-AEVB). MADE adapted from https://github.com/karpathy/pytorch-made,
IAF from https://github.com/pclucas14/iaf-vae.'''
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(torch.from_numpy(mask.astype(np.uint8).T))

    def forward(self, input):
        return F.linear(input, self.mask * self.weight, self.bias)


class MADE(nn.Module):
    '''Masked autoregressive MLP; the 10-dimensional context h is added after the first layer.'''
    def __init__(self, nin, hidden_sizes, nout, context_size=10):
        super().__init__()
        assert nout % nin == 0, 'nout must be integer multiple of nin'
        self.nin, self.nout, self.hidden_sizes = nin, nout, hidden_sizes
        self.h_layer = nn.Linear(context_size, hidden_sizes[0])
        net = []
        hs = [nin] + hidden_sizes + [nout]
        for h0, h1 in zip(hs, hs[1:]):
            net.extend([MaskedLinear(h0, h1), nn.ReLU()])
        net.pop()
        self.net = nn.Sequential(*net)
        self._set_masks()

    def _set_masks(self):
        L = len(self.hidden_sizes)
        rng = np.random.RandomState(0)
        m = {-1: rng.permutation(self.nin)}
        for l in range(L):
            m[l] = rng.randint(m[l - 1].min(), self.nin - 1, size=self.hidden_sizes[l])
        masks = [m[l - 1][:, None] <= m[l][None, :] for l in range(L)]
        masks.append(m[L - 1][:, None] < m[-1][None, :])
        if self.nout > self.nin:
            masks[-1] = np.concatenate([masks[-1]] * (self.nout // self.nin), axis=1)
        for layer, mask in zip([l for l in self.net.modules() if isinstance(l, MaskedLinear)], masks):
            layer.set_mask(mask)

    def forward(self, x, h):
        for i, layer in enumerate(self.net):
            x = layer(x)
            if i == 0:
                x = x + self.h_layer(h)
        return x


class ARMLP(nn.Module):
    def __init__(self, nin, nout, nh):
        super().__init__()
        self.net = MADE(nin, [nh, nh, nh], nout)

    def forward(self, x, h):
        return self.net(x, h)


class IAF(nn.Module):
    '''x' = s * z + (1 - s) * t with (s, t) autoregressive in z; log |det| = sum log s. parity flips the order.'''
    def __init__(self, dim_z, parity, nh=128):
        super().__init__()
        self.dim = dim_z
        self.net = ARMLP(dim_z, dim_z * 2, nh)
        self.parity = parity

    def forward(self, z, h):
        s, t = self.net(z, h).split(self.dim, dim=1)
        var = torch.sigmoid(s)
        log_var = torch.log(var)
        x = var * z + (1 - var) * t
        x = x.flip(dims=(1,)) if self.parity else x
        return x, torch.sum(log_var, dim=1)
