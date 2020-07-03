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


def train_and_test(model, device, train_loader, optimizer, saver, writer,
                   num_epochs = 20, loss_func = F.nll_loss):

    for epoch in range(1, num_epochs + 1):
        print("starting epoch {} of {}".format(epoch, num_epochs))
        train(model, device, train_loader, optimizer, epoch,
              loss_func = loss_func, writer = writer)
        loss, acc, data, labels, preds, target = test(model, device, test_loader,
                                                      epoch, writer = writer)
        saver.save(model, optimizer, epoch, loss, acc)

    return model, loss, acc, data, labels, preds, target


if __name__ == "__main__":
    device = torch.device("cuda")
    timestamp = generate_timestamp()
    writer = SummaryWriter()
    dataset_name = "ImageNet"
    batch_size = 64
    num_epochs = 4
    lr = 1e-3
    data_folder = "./data"
    train_loader,\
    test_loader,\
    image_size = load_dataset(dataset_name, batch_size, data_folder)
    in_ch = image_size[0]
    shape = image_size[1:]
    flat_shape = np.prod(image_size)
    f1 = nn.Sequential(Conv2d_pad(in_ch, 8, 3), nn.MaxPool2d(2),
                       Conv2d_pad(8, in_ch, 3), nn.UpsamplingBilinear2d(shape))
    f2 = nn.Sequential(Conv2d_pad(in_ch, 8, 3),nn.ReLU(),
                       Conv2d_pad(8, in_ch, 3), nn.ReLU())
    f3 = nn.Sequential(Conv2d_pad(in_ch, 8, 5), nn.MaxPool2d(2),
                       Conv2d_pad(8, in_ch, 5), nn.UpsamplingBilinear2d(shape))
    f4 = nn.Sequential(Conv2d_pad(in_ch, 8, 5), nn.ReLU(),
                       Conv2d_pad(8, in_ch, 5), nn.ReLU())
    f5 = nn.Sequential(Conv2d_pad(in_ch, 8, 7), nn.MaxPool2d(2),
                       Conv2d_pad(8, in_ch, 7), nn.UpsamplingBilinear2d(shape))
    f6 = nn.Sequential(Conv2d_pad(in_ch, 8, 7), nn.ReLU(),
                       Conv2d_pad(8, in_ch, 7), nn.ReLU())
    funcs = [f1, f2, f3, f4,  f5, f6]
    kernels = [TemporalKernel(torch.rand(10).to(device),
                              torch.rand(10).to(device),
                              torch.rand(10).to(device))
               for _ in funcs]
    t = torch.tensor([0, 1]).float().to(device)
    classifier = nn.Sequential(*[nn.Linear(flat_shape, 10)])
    model = VolterraClassifier(funcs, kernels, t, classifier)
    optimizer = torch.optim.Adam(model.parameters(), lr= lr)
    model = model.to(device)
    writer = None
    model_path = os.path.join('./checkpoints/conv_' + dataset_name, timestamp)
    saver = BestModelSaver(model_path)
    model, loss, acc, data,\
    labels, preds, target = train_and_test(model, device, train_loader, optimizer,
                                           saver, writer, num_epochs = num_epochs,
                                           loss_func = F.nll_loss)

    f1 = nn.Sequential(Conv2d_pad(in_ch, 3, 3), nn.MaxPool2d(2),
                       Conv2d_pad(3, in_ch, 3), nn.UpsamplingBilinear2d(shape))
    f2 = nn.Sequential(Conv2d_pad(in_ch, 3, 3),nn.ReLU(),
                       Conv2d_pad(3, in_ch, 3), nn.ReLU())
    f3 = nn.Sequential(Conv2d_pad(in_ch, 3, 5), nn.MaxPool2d(2),
                       Conv2d_pad(3, in_ch, 3), nn.UpsamplingBilinear2d(shape))
    f4 = nn.Sequential(Conv2d_pad(in_ch, 3, 3), nn.ReLU(),
                       Conv2d_pad(3, in_ch, 5), nn.ReLU())
    f5 = nn.Sequential(Conv2d_pad(in_ch, 8, 3), nn.MaxPool2d(2),
                       Conv2d_pad(8, in_ch, 3), nn.UpsamplingBilinear2d(shape))
    f6 = nn.Sequential(Conv2d_pad(in_ch, 8, 3), nn.ReLU(),
                       Conv2d_pad(8, in_ch, 3), nn.ReLU())
    funcs = [f1, f2, f3, f4,  f5, f6]
    val_classifier = nn.Sequential(*[nn.Linear(flat_shape, 10)])
    val_model = ParallelModel(funcs, val_classifier)
    val_model = val_model.to(device)
    val_optimizer = torch.optim.Adam(val_model.parameters(), lr= lr)
    val_writer = None
    val_model_path = os.path.join('./checkpoints/validation_' + dataset_name, timestamp)
    val_saver = BestModelSaver(val_model_path)
    val_model, val_loss, val_acc, val_data,\
    val_labels, val_preds, val_target = train_and_test(val_model, device,
                                                       train_loader,
                                                       val_optimizer,
                                                       val_saver, val_writer,
                                                       num_epochs = num_epochs,
                                                       loss_func = F.nll_loss)
