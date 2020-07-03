import os
import sys
sys.path.append("../")
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("paper")
sns.set_palette("colorblind")


def train_and_test(model, device, train_loader, test_loader, optimizer, saver,
                   writer, num_epochs = 20, cost = 0.01):
    history = {"train_loss":[], "train_acc":[], "test_loss":[], "test_acc":[]}

    for epoch in range(1, num_epochs + 1):
        print("starting epoch {} of {}".format(epoch, num_epochs))
        train_loss, train_acc = train(model, device, train_loader, optimizer,
                                      epoch, coeff = cost, writer = writer)
        test_loss, test_acc = test(model, device, test_loader, epoch,
                                   coeff = cost, writer = writer)
        saver.save(model, optimizer, epoch, test_loss, test_acc)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)

    return model, history

"""Training wrappers"""
def reg_loss(output, target, reg = 0, coeff = 0., func = nn.CrossEntropyLoss()):
    return func(output, target) + coeff * reg


def train(model, device, train_loader, optimizer, epoch,
          loss_func = reg_loss, coeff = 0., writer = None):
    steps = 100
    correct = 0
    n_total = 0
    model.train()
    epoch_loss = []
    epoch_acc = []
    emb_mat = []
    emb_meta = []

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, reg = model(data)
        l1 = loss_func(output, target, reg = reg, coeff = coeff)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        n_total += data.shape[0]
        l1.backward()
        optimizer.step()
        if writer is not None:
            writer.add_scalar('Loss/train', l1.item(), epoch)
            writer.add_scalar('Acc/train', 100. * correct / n_total, epoch)
            writer.add_histogram('c_s/train', model.learning_machine.odefunc.c_s,
                                 epoch)


            # for time in torch.linspace(0, 1, steps=5):
                # images = model.learning_machine.get_images_at(time, device, data, target)

                # for label, output in zip(target, images):
                    # for i in range(output.shape[0]):
                    # ch = 0
                    # if label == 0:
                        # writer.add_images('samples/train' + "_time{}".format(time) +
                                          # "/label_{}_channel_{}".format(label, ch),
                                          # output[ch:ch+1, :,:].unsqueeze(0),epoch)

                # writer.add_embedding(images.reshape([data.shape[0], -1]),
                #                      metadata=target, global_step=epoch,
                #                      tag=str(time))

        epoch_loss.append(l1.item())
        epoch_acc.append(correct / n_total)
        if True:#batch_idx % steps == 0 and batch_idx != 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc {:.3f}%'.format(
                epoch, (batch_idx+1) * len(data), len(train_loader.dataset),
                100. * (batch_idx+1) / len(train_loader), l1.item(),
                100. * correct / n_total))


    return np.mean(epoch_loss), np.mean(epoch_acc)


def test(model, device, test_loader, epoch, loss_func = reg_loss, coeff = 0.,
         writer = None):
    model.eval()
    test_loss = 0
    correct = 0
    preds = []
    targets = []
    epoch_loss = []
    epoch_acc = []
    with torch.no_grad():

        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, reg = model(data)
            test_loss += loss_func(output, target, reg = reg, coeff = coeff).item()
            pred = output.argmax(dim=1, keepdim=True)
            preds.append(pred)
            targets.append(target)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    epoch_loss.append(test_loss)
    epoch_acc.append(correct / len(test_loader.dataset))
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.3f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_acc))
    if writer is not None:
        writer.add_scalar('Loss/val', test_loss, epoch)
        writer.add_scalar('Acc/val', test_acc, epoch)
    return np.mean(epoch_loss), np.mean(epoch_acc)


class BestModelSaver:
    def __init__(self, path):
        self.loss = 1e+6
        self.check_path(path)
        self.path = os.path.join(path, 'checkpoint.pth.tar')
        self.best_path = os.path.join(path, 'model_best.pth.tar')

    @staticmethod
    def check_path(path):
        if not os.path.isdir(path):
            os.makedirs(path)


    def is_best(self, new_loss):
        return self.loss > new_loss


    def save(self, model, optimizer, epoch, loss, acc):
        state = {
            'epoch': epoch,
            'loss': loss,
            'state_dict': model.state_dict(),
            'best_acc1': acc,
            'optimizer' : optimizer.state_dict(),
        }
        torch.save(state, self.path)
        if self.is_best(loss):
            self.loss = loss
            shutil.copyfile(self.path, self.best_path)


def select_classes(dataset, classes):
    inds = np.zeros(dataset.targets.shape).astype(np.bool)
    for c in classes:
        inds[(dataset.targets == c)] = True
    return inds


def select_number_of_samples(dataset, num_samples):
    sample_inds = []

    for c in np.unique(dataset.targets):
        temp = np.where(dataset.targets == c)[0]
        np.random.seed(0)
        p = np.random.permutation(temp.shape[0])
        sample_inds.extend(temp[p][:num_samples])

    inds = np.zeros(dataset.targets.shape).astype(np.bool)
    inds[sample_inds] = True
    return inds


def load_dataset(name, batch_size, data_folder = "./data", classes = None,
                 num_sample_per_class = None):
    if name == "MNIST":
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))
                                        ])
        dataset = datasets.MNIST(data_folder, train=True, download=True,
                                 transform=transform)
        if classes is not None:
            inds = select_classes(dataset, classes)
            dataset = torch.utils.data.Subset(dataset, indices=np.where(inds == True)[0])
            train_loader = torch.utils.data.DataLoader(dataset,
                                drop_last=False, batch_size=batch_size, shuffle=True)
        else:
            train_loader = torch.utils.data.DataLoader(dataset,
                                drop_last=False, batch_size=batch_size, shuffle=True)

        if num_sample_per_class is not None:
            inds = select_number_of_samples(dataset, num_sample_per_class)
            dataset = torch.utils.data.Subset(dataset, indices=np.where(inds == True)[0])
            train_loader = torch.utils.data.DataLoader(dataset,
                                drop_last=False, batch_size=batch_size, shuffle=True)



        dataset = datasets.MNIST(data_folder, train=False, download=True,
                                 transform=transform)
        if classes is not None:
            inds = select_classes(dataset, classes)
            dataset = torch.utils.data.Subset(dataset, np.where(inds == True)[0])
            test_loader = torch.utils.data.DataLoader(dataset,
                               drop_last=False, batch_size=batch_size, shuffle=True)
        else:
            test_loader = torch.utils.data.DataLoader(dataset,
                               drop_last=False, batch_size=batch_size, shuffle=True)
            classes = [i for i in range(10)]

        image_size = (1,28,28)
    else:
        raise NotImplementedError("The dataset is not available")
    return train_loader, test_loader, image_size, classes
