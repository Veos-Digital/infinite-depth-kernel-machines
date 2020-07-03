from math import tau
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torchdiffeq import odeint_adjoint as odeint

def squash_in_sphere(v):
    norm = torch.sqrt(torch.sum(v * v, dim=1, keepdim=True) + 1e-6)
    scale = torch.sigmoid(norm) / norm
    return v * scale


def adjustedkernel(u, v):
    uu = torch.sum(u * u, dim=1, keepdim=True)
    vv = torch.sum(v * v, dim=1, keepdim=True)
    uv = torch.matmul(u, torch.t(v))
    res = uv - uu / 2 - torch.t(vv) / 2
    return torch.exp(res) * uv


def kernel(u, v):
    u, v = squash_in_sphere(u), squash_in_sphere(v)
    return adjustedkernel(u, v)


class CKFunction(nn.Module):
    def __init__(self, psi_s, num_samples, c_s):
        super(CKFunction, self).__init__()
        self.psi_s = psi_s
        self.num_samples = num_samples
        self.c_s = nn.Parameter(c_s)
        max_freq, device, dtype = c_s.shape[-1], c_s.device, c_s.dtype
        self._coeffs = tau * torch.arange(max_freq, device=device, dtype=dtype)

    def get_weights(self, time):
        return torch.cos(time * self._coeffs)

    def forward(self, time, z):
        weights = self.get_weights(time)
        c = torch.matmul(self.c_s, weights)
        psi = torch.matmul(self.psi_s, weights) # mb_cat, num_features
        u = psi + torch.matmul(z, c)
        return kernel(u, u[0:self.num_samples, :]) # mb_cat, num_samples


def flat_dot(u, v):
    return torch.dot(u.view(-1), v.view(-1))

class CKMachine(nn.Module):
    def __init__(self, data, aug_func = None,
                 num_aug_channels = 8, max_freq = 10, learn_data = False):
        super(CKMachine, self).__init__()
        self.num_aug_channels = num_aug_channels
        self.max_freq = max_freq
        self.data = data
        self.data.requires_grad = learn_data
        device, dtype = self.data.device, self.data.dtype
        self.interval = torch.tensor(torch.linspace(0, 1, steps=max_freq),
                                     device=device, dtype=dtype)
        self.num_samples = data.shape[0]
        # TODO: num_aug_channels deve tenere conto dell'architettura dell'augmenter (13*13)
        c_s = torch.empty(self.num_samples, 13*13*num_aug_channels,
                          max_freq, device=device, dtype=dtype)
        xavier_uniform_(c_s)
        self.psi = Augmenter(transformation = aug_func,
                             num_aug_channels = num_aug_channels,
                             max_freq = max_freq)
        self.odefunc = CKFunction(None, self.num_samples, c_s)

    def forward(self, inputs):
        psi_s, cost = self.psi(torch.cat([self.data, inputs], dim=0))
        self.odefunc.psi_s = psi_s
        mb_cat = psi_s.shape[0]
        z_shape = (mb_cat, self.num_samples)
        z = torch.zeros(z_shape, device = psi_s.device, dtype = psi_s.dtype)
        interval = self.interval
        zs = odeint(self.odefunc, z, interval, rtol=1e-3, atol=1e-3)
        for z, time in zip(zs, interval):
            c = torch.matmul(self.odefunc.c_s, self.odefunc.get_weights(time))
            cost += flat_dot(torch.matmul(z[0:self.num_samples, :], c), c) / self.max_freq

        return torch.cat(list(zs), dim=1)[self.num_samples:, :], cost


class CKClassifier(nn.Module):
    def __init__(self, data, classifier, num_aug_channels = 8,
                 max_freq = 10, learn_data = False):
        super(CKClassifier, self).__init__()
        self.learning_machine = CKMachine(data,
                                          num_aug_channels = num_aug_channels,
                                          max_freq = max_freq,
                                          learn_data = learn_data)
        self.classifier = classifier

    def forward(self, inputs):
        x, reg = self.learning_machine(inputs)
        x = x.view(x.shape[0], -1)
        return self.classifier(x), reg + square_norm(self.classifier)


class Augmenter(nn.Module):
    def __init__(self, transformation=None, num_aug_channels = 8, max_freq = 10):
        super(Augmenter, self).__init__()
        self.num_aug_channels = num_aug_channels
        self.max_freq = max_freq
        self.transformation = nn.Conv2d(1, num_aug_channels * max_freq, 3, 2)

    def forward(self, inputs):
        shape = (inputs.shape[0], self.max_freq, -1)
        res = torch.reshape(self.transformation(inputs), shape)
        return torch.transpose(res, 1, 2), square_norm(self)


def square_norm(model):
    cost = 0

    for param in model.parameters():
        if param.requires_grad:
            cost += torch.sum(param * param)

    return cost
