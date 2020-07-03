from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
from utils import Conv2d_pad

class VolterraFunction(nn.Module):
    def __init__(self, funcs, kernels, psi, inputs):
        super(VolterraFunction, self).__init__()
        self.funcs = funcs
        self.kernels = kernels
        self.inputs = inputs
        self.psi = psi

    def getu(self, t, ys):
        acc = self.psi(t, self.inputs)
        for ker, y in zip(self.kernels, ys):
            acc += ker(t, y)
        return acc

    def forward(self, t, ys):
        u = self.getu(t, ys)
        ysnew = [f(t, u) for f in self.funcs]
        return torch.stack(ysnew)


class VolterraMachine(nn.Module):
    def __init__(self, funcs, kernels, interval, psi):
        super(VolterraMachine, self).__init__()
        self.funcs = nn.ModuleList(funcs)
        self.kernels = nn.ModuleList(kernels)
        self.interval = interval
        self.psi = psi

    def forward(self, inputs):
        odefunc = VolterraFunction(self.funcs, self.kernels, self.psi, inputs)
        psi0 = self.psi(0, inputs)
        ys = torch.stack([torch.zeros_like(psi0) for _ in self.funcs])
        interval = self.interval.type_as(psi0)
        ys_final = odeint(odefunc, ys, interval, rtol=1e-3, atol=1e-3)[-1]
        return odefunc.getu(interval[1], ys_final)


class VolterraClassifier(nn.Module):
    def __init__(self, funcs, kernels, interval, psi, classifier):
        super(VolterraClassifier, self).__init__()
        self.learning_machine = VolterraMachine(funcs, kernels, interval, psi)
        self.classifier = classifier

    def forward(self, inputs):
        x = self.learning_machine(inputs)
        shape = torch.tensor(x.shape[0]).item()
        x = x.view(shape, -1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    fs1 = [torch.nn.Sequential(
        torch.nn.ReLU(),
        torch.nn.Conv2d(3, 3, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(3, 3, 3, padding=1),
    ) for _ in range(10)]
    fs2 = [torch.nn.Sequential(
        torch.nn.ReLU(),
        torch.nn.Conv2d(3, 3, 3, padding=1),
        torch.nn.ReLU(),
        torch.nn.Conv2d(3, 3, 3, padding=1),
    ) for _ in range(10)]
    f1 = TemporalModule(fs1, torch.rand(10), torch.rand(10))
    f2 = TemporalModule(fs2, torch.rand(10), torch.rand(10))
    funcs = [f1, f2]

    temporal_conv1 = [Conv2d_pad(3,3,3) for _ in range(10)]
    kernel1 = TemporalModule(temporal_conv1, torch.rand(10), torch.rand(10))
    temporal_conv2 = [Conv2d_pad(3,3,3) for _ in range(10)]
    kernel2 = TemporalModule(temporal_conv2, torch.rand(10), torch.rand(10))
    kernels = [kernel1, kernel2]

    t = torch.tensor([0, 1]).float()
    psiconv = nn.ModuleList([Conv2d_pad(3,3,3) for _ in range(10)])
    psi = TemporalModule(psiconv, torch.rand(10), torch.rand(10))
    volt = VolterraMachine(funcs, kernels, t, psi)

    volt(torch.rand(5, 3, 5, 5))
