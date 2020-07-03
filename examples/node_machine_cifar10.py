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
from node_machine import NodeClassifier
from evaluate_on_dataset import train_and_test
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)

if __name__ == "__main__":
    device = torch.device("cuda")
    timestamp = generate_timestamp()
    writer = SummaryWriter()
    dataset_name = "MNIST"
    batch_size, num_epochs, lr = 32, 20, 1e-4
    data_folder = "../data"
    train_loader,\
    test_loader,\
    image_size, classes = load_dataset(dataset_name, batch_size, data_folder)
    in_ch = image_size[0]
    shape = image_size[1:]
    flat_shape = np.prod(image_size)
    ch_in, ch_out = 8, 512
    p_h, p_w = 3, 3
    x_s = torch.rand(ch_out, ch_in, p_h, p_w).to(device)
    c_s = torch.rand(ch_in, ch_out, 1, 1).to(device)
    lambdas = torch.rand(ch_out).to(device)
    times = torch.rand(ch_out).to(device)
    interval = torch.tensor([0, 1]).float()
    classifier = nn.Sequential(*[nn.Linear(288, 10)])
    model = NodeClassifier(x_s, c_s, lambdas, times, interval, classifier)
    optimizer = torch.optim.Adam(model.parameters(), lr= lr)
    model = model.to(device)
    writer = None
    model_path = os.path.join('./checkpoints/node_' + dataset_name, timestamp)
    saver = BestModelSaver(model_path)
    model, history = train_and_test(model, device, train_loader, test_loader, optimizer,
                                    saver, writer, num_epochs = num_epochs,
                                    loss_func = nn.CrossEntropyLoss())
