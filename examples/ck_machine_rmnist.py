import os
import sys
sys.path.append("../")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from utils import generate_timestamp
from continuous_kernel_machine import CKClassifier
from evaluate_on_dataset import train_and_test, load_dataset, BestModelSaver
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(0)
import matplotlib.pyplot as plt
from matplotlib import cm
from natsort import natsorted
import pandas as pd
import joypy
import seaborn as sns
sns.set_context("paper")
sns.set_palette("colorblind")


def train_and_test_on_rmnist(num_sample_per_class, writer, num_aug_channels, lr,
                             cost = 0.01, max_freq = 10):
    train_loader,\
    test_loader,\
    image_size, classes = load_dataset(dataset_name, batch_size, data_folder,
                                       num_sample_per_class = num_sample_per_class)

    for data, label in train_loader:
        data, label = data.to(device), label.to(device)

    classifier = nn.Sequential(*[nn.Linear(data.shape[0]*max_freq, 10)])

    model = CKClassifier(data, classifier,
                         num_aug_channels = num_aug_channels,
                         max_freq = max_freq, learn_data = False)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr= lr, weight_decay=0.01)
    dummy_input = torch.rand((10,) +  (1,28,28)).to(device)
    writer.add_graph(model=model, input_to_model=(dummy_input, ))
    name = "{}{}_mf{}_c{}".format(num_sample_per_class, dataset_name, max_freq, cost)
    model_path = os.path.join('./checkpoints/ck_mnist' + name, timestamp)
    saver = BestModelSaver(model_path)
    model, history = train_and_test(model, device, train_loader, test_loader,
                                    optimizer, saver, writer, num_epochs = num_epochs,
                                    cost = cost)
    return model, history


def plot_test_losses_and_accs(histories, dataset_name = "MNIST",
                num_samples = ["1", "5", "10"]):
    f, axs = plt.subplots(1, 2)
    values = ["acc", "loss"]

    for ax, v in zip(axs, values):
        for n, h in zip(num_samples, histories):
            ax.plot(h["test_"+v], label=n)

    ax.legend()
    color = tensor2numpy(model.learning_machine.times)
    size = tensor2numpy(model.learning_machine.lambdas)


if __name__ == "__main__":
    device = torch.device("cuda")
    timestamp = generate_timestamp()

    dataset_name = "MNIST"
    num_classess = 10
    batch_size, num_epochs = 128, 10
    data_folder = "../data"
    num_aug_channels, lr = 8, .04
    hists = []
    models = []
    max_freqs = [2,5,10,20]
    cost = 0.1#[0, 0.001, 0.01, 0.1, 1]

    for max_freq in max_freqs:

        for num_sample_per_class in [1]:#, 5, 10]:
            writer = SummaryWriter()
            model, history = train_and_test_on_rmnist(num_sample_per_class,
                                                      writer, num_aug_channels,
                                                      lr, cost=cost,
                                                      max_freq=max_freq) # forse anche piu?
            hists.append(history)
            models.append(model)


def plot_cs_histograms(path_to_events, ax = None):
    event_acc = EventAccumulator(path_to_events, size_guidance={
    'histograms': 10,
    })
    event_acc.Reload()
    tags = event_acc.Tags()
    result = {}
    for hist in tags['histograms']:
        histograms = event_acc.Histograms(hist)
        to_plot = np.array([np.repeat(np.array(h.histogram_value.bucket_limit),
                                           np.array(h.histogram_value.bucket).astype(np.int))
                                 for h in histograms])
    df = pd.DataFrame(to_plot.T)
    ax = joypy.joyplot(df, overlap=2, colormap=cm.OrRd_r, linecolor='w', linewidth=.5,
                  ax = ax)
    return result


def plot_history(histories, key, legend, ax = None):
    if ax is None: f,ax = plt.subplots()

    for i, (h, l) in enumerate(zip(histories, legend)):
        ax.plot(h[key], label = l)
        if i == 0:
            ax.set_title(key)

    ax.legend()
    sns.despine()

f, axs = plt.subplots(2,2)
axs = axs.ravel()
keys = ["train_loss", "train_acc", "test_loss","test_acc"]

for i, key in enumerate(keys):
    plot_history(hists, key, max_freqs, ax = axs[i])

plt.show()

folder = './runs/varying_max_freq'
hist_paths = natsorted([ os.path.join(folder, f) for f in os.listdir(folder)])

for i, path in enumerate(hist_paths):
    f = plot_cs_histograms(path)
    plt.savefig("mf{}_c{}.pdf".format(max_freqs[i], cost))
