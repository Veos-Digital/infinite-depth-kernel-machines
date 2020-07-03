import torch
import torch.nn as nn
import torch.nn.functional as F
from continuous_kernel_machine import Augmenter
from torchdiffeq import odeint

class NodeFunction(nn.Module):
    def __init__(self, psi_s, x_s, c_s, lambdas, times):
        super(NodeFunction, self).__init__()
        self.psi_s = psi_s
        self.x_s = x_s
        self.c_s = c_s
        self.lambdas = lambdas
        self.times = times

    def get_weights(self, time):
        return torch.exp(-torch.pow(self.lambdas * (self.times - time), 2))

    def get_x_c(self, time):
        weights = self.get_weights(time)
        c = self.c_s * torch.reshape(weights, (1, -1, 1, 1))
        x = self.x_s * torch.reshape(weights, (-1, 1, 1, 1))
        return x, c

    def get_u(self, time, diff):
        psi = self.psi_s
        return psi + diff

    def forward(self, time, diff):
        x, c = self.get_x_c(time)
        u = self.get_u(time, diff)
        z = equivariant_radialkernel(u, x)
        return F.conv2d(z, c)


class NodeMachine(nn.Module):
    def __init__(self, x_s, c_s, lambdas, times, interval, aug_func = None):
        super(NodeMachine, self).__init__()
        self.x_s = x_s
        self.c_s = c_s
        self.lambdas = lambdas
        self.times = times
        self.interval = interval
        self.psi = Augmenter(self.times, transformation = aug_func)

    def forward(self, inputs):
        psi_s = self.psi(inputs)
        odefunc = NodeFunction(psi_s, self.x_s, self.c_s, self.lambdas, self.times)
        diff = torch.zeros_like(psi_s)
        interval = self.interval
        diff = odeint(odefunc, diff, interval, rtol=1e-3, atol=1e-3)[-1]
        return odefunc.get_u(interval[1], diff)


class NodeClassifier(nn.Module):
    def __init__(self, x_s, c_s, lambdas, times, interval, classifier):
        super(NodeClassifier, self).__init__()
        self.learning_machine = NodeMachine(x_s, c_s, lambdas, times, interval)
        self.classifier = classifier

    def forward(self, inputs):
        x = self.learning_machine(inputs)
        shape = torch.tensor(x.shape[0]).item()
        x = x.view(shape, -1)
        return self.classifier(x)
