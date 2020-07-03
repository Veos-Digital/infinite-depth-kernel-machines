import os
import sys
sys.path.append("../")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from utils import load_dataset, train, test, generate_timestamp, Conv2d_pad, BestModelSaver
from utils import SerialModel, ParallelModel
from volterra_machine import TemporalModule, VolterraMachine, VolterraClassifier
from evaluate_on_dataset import train_and_test
torch.autograd.set_detect_anomaly(True)


if __name__ == "__main__":
    device = torch.device("cuda")
    timestamp = generate_timestamp()
    writer = SummaryWriter()
    dataset_name = "CIFAR10"
    batch_size = 64
    num_epochs = 10
    lr = 1e-3
    data_folder = "../data"
    train_loader,\
    test_loader,\
    image_size, classes = load_dataset(dataset_name, batch_size, data_folder)
    in_ch = image_size[0]
    shape = image_size[1:]
    flat_shape = np.prod(image_size)
    fs1 = [nn.Sequential(
        nn.ReLU(),
        Conv2d_pad(3, 3, 3),
        nn.ReLU(),
        Conv2d_pad(3, 3, 3),
    ) for _ in range(10)]
    fs2 = [nn.Sequential(
        nn.ReLU(),
        Conv2d_pad(3, 3, 3),
        nn.ReLU(),
        Conv2d_pad(3, 3, 3),
    ) for _ in range(10)]
    temporal_acc_f1 = nn.Sequential(
        nn.ReLU(),
        Conv2d_pad(3, 3, 3),
        nn.ReLU(),
        Conv2d_pad(3, 3, 3))
    temporal_acc_f2 = nn.Sequential(
        nn.ReLU(),
        Conv2d_pad(3, 3, 3),
        nn.ReLU(),
        Conv2d_pad(3, 3, 3))
    f1 = TemporalModule(fs1, torch.rand(10).to(device),
                        torch.rand(10).to(device), temporal_acc_f1)
    f2 = TemporalModule(fs2, torch.rand(10).to(device),
                        torch.rand(10).to(device), temporal_acc_f2)
    funcs = [f1, f2]

    temporal_conv1 = [Conv2d_pad(3,3,3) for _ in range(10)]
    temporal_acc1 = Conv2d_pad(3,3,3)
    kernel1 = TemporalModule(temporal_conv1, torch.rand(10).to(device),
                             torch.rand(10).to(device), temporal_acc1)
    temporal_conv2 = [Conv2d_pad(3,3,3) for _ in range(10)]
    temporal_acc2 = Conv2d_pad(3,3,3)
    kernel2 = TemporalModule(temporal_conv2, torch.rand(10).to(device),
                             torch.rand(10).to(device), temporal_acc2)
    kernels = [kernel1, kernel2]

    t = torch.tensor([0, 1]).float()
    psiconv = nn.ModuleList([Conv2d_pad(3,3,3) for _ in range(10)])
    temporal_acc_psi = Conv2d_pad(3,3,3)
    psi = TemporalModule(psiconv, torch.rand(10).to(device),
                         torch.rand(10).to(device), temporal_acc_psi)
    classifier = nn.Sequential(*[nn.Linear(flat_shape, 10)])
    model = VolterraClassifier(funcs, kernels, t, psi, classifier)
    optimizer = torch.optim.Adam(model.parameters(), lr= lr)
    model = model.to(device)
    writer = None
    model_path = os.path.join('./checkpoints/conv_' + dataset_name, timestamp)
    saver = BestModelSaver(model_path)
    model, history = train_and_test(model, device, train_loader, test_loader, optimizer,
                                    saver, writer, num_epochs = num_epochs,
                                    loss_func = F.nll_loss)

    # f1 = nn.Sequential(Conv2d_pad(in_ch, 3, 3), nn.MaxPool2d(2),
    #                    Conv2d_pad(3, in_ch, 3), nn.UpsamplingBilinear2d(shape))
    # f2 = nn.Sequential(Conv2d_pad(in_ch, 3, 3),nn.ReLU(),
    #                    Conv2d_pad(3, in_ch, 3), nn.ReLU())
    # f3 = nn.Sequential(Conv2d_pad(in_ch, 3, 5), nn.MaxPool2d(2),
    #                    Conv2d_pad(3, in_ch, 3), nn.UpsamplingBilinear2d(shape))
    # f4 = nn.Sequential(Conv2d_pad(in_ch, 3, 3), nn.ReLU(),
    #                    Conv2d_pad(3, in_ch, 5), nn.ReLU())
    # f5 = nn.Sequential(Conv2d_pad(in_ch, 8, 3), nn.MaxPool2d(2),
    #                    Conv2d_pad(8, in_ch, 3), nn.UpsamplingBilinear2d(shape))
    # f6 = nn.Sequential(Conv2d_pad(in_ch, 8, 3), nn.ReLU(),
    #                    Conv2d_pad(8, in_ch, 3), nn.ReLU())
    # funcs = [f1, f2, f3, f4,  f5, f6]
    # val_classifier = nn.Sequential(*[nn.Linear(flat_shape, 10)])
    # val_model = ParallelModel(funcs, val_classifier)
    # val_model = val_model.to(device)
    # val_optimizer = torch.optim.Adam(val_model.parameters(), lr= lr)
    # val_writer = None
    # val_model_path = os.path.join('./checkpoints/validation_' + dataset_name, timestamp)
    # val_saver = BestModelSaver(val_model_path)
    # val_model, val_history = train_and_test(val_model, device, train_loader,
    #                                         val_optimizer, val_saver, val_writer,
    #                                         num_epochs = num_epochs,
    #                                         loss_func = F.nll_loss)
