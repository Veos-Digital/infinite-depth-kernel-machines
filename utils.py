import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)
from math import floor
import numpy as np
import matplotlib
from matplotlib import cm
from datetime import datetime
import json
import umap


def generate_timestamp():
    return datetime.now().isoformat()[:-7].replace("T","-").replace(":","-")

"""Pytorch wrappers"""
def compute_out_shape(size, layer):
    if is_conv_like(layer):
        h_out, w_out = conv_out_shape(size, layer)
    elif is_upsampling:
        h_out, w_out = upsampling_out_shape(size, layer)
    else:
        print(layer)
        message = "Please implement a function to return out shape"
        raise NotImplementedError(message)
    return h_out, w_out


def conv_out_shape(size, l):
    from math import floor
    p = get_conv_like_attrs(l.padding)
    d = get_conv_like_attrs(l.dilation)
    k = get_conv_like_attrs(l.kernel_size)
    s = get_conv_like_attrs(l.stride)
    h = floor(((size[0] + (2 * p[0]) - d[0] * (k[0] - 1) - 1 ) / s[0]) + 1)
    w = floor(((size[1] + (2 * p[1]) - d[1] * (k[1] - 1) - 1 ) / s[1]) + 1)
    return h, w


def upsampling_out_shape(size, l):
    if l.scale_factor is not None:
        return size[0] * l.scale_factor, size[1] * l.scale_factor
    elif l.size is not None:
        s = get_conv_like_attrs(l.size)
        return size[0] * s[0], size[1] * s[1]


def get_conv_like_attrs(attr):
    if isinstance(attr, int):
        attr  = [attr, attr]
    elif len(attr) == 1:
        attr = [attr[0], attr[0]]
    return attr


def is_upsampling(l):
    return (isinstance(layer, nn.Upsample) or
            isinstance(layer, nn.UpsamplingNearest2d) or
            isinstance(layer, nn.UpsamplingBilinear2d))


def is_conv_like(l):
    return (isinstance(l, nn.Conv2d) or
            isinstance(l, nn.MaxPool2d)  or
            isinstance(l, IeneoLayer))


'''Visualization
'''
#adapted from https://umap-learn.readthedocs.io/en/latest/parameters.html
def draw_umap(data, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean',
              title='', color = None, alpha = 0., size = None, f = None, ax = None):
    u = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    emb = u.fit_transform(data)
    if size is None: size = 100
    if color is None: color = data
    if ax is None: f = plt.figure()
    if n_components == 1:
        sc = ax.scatter(emb[:,0], np.arange(emb.shape[0]), c = color, s = size,
                   alpha = alpha)
    if n_components == 2:
        sc = ax.scatter(emb[:,0], emb[:,1], c = color, s = size, alpha = alpha)
    if n_components == 3:
        ax = f.add_subplot(111, projection='3d')
        sc = ax.scatter(emb[:,0], emb[:,1], emb[:,2], c = color, s = size, alpha = alpha)
    plt.title(title)
    if size is not None:
        handles, labels = sc.legend_elements(prop="sizes", alpha=0.6)
        legend2 = ax.legend(handles, labels, loc="upper right", title="Lambdas",
                            fancybox=True, framealpha=0.5)
        ax.add_artist(legend2)
    if color is not None:
        legend1 = ax.legend(*sc.legend_elements(),
                    loc="lower left", title="Times", fancybox=True, framealpha=0.5)
        ax.add_artist(legend1)


def tensor2numpy(pytorch_tensor):
    return pytorch_tensor.detach().cpu().numpy()


def get_embedding_data(model, device):
    model.eval()
    data = tensor2numpy(model.learning_machine.x_s)
    data2 = tensor2numpy(model.learning_machine.c_s).T
    color = tensor2numpy(model.learning_machine.times)
    size = tensor2numpy(model.learning_machine.lambdas)
    data = (data - data.mean()) / data.std()
    data2 = (data2 - data2.mean()) / data2.std()
    f, axs = plt.subplots(1,2)
    draw_umap(data, n_neighbors=2, min_dist=0.5, n_components=2,
              metric='euclidean', title='x_s', color = color, alpha = .6,
              size = 100*size, f = f, ax = axs[0])
    draw_umap(data2, n_neighbors=2, min_dist=0.5, n_components=2,
              metric='euclidean', title='c_s', color = color, alpha = .6,
              size = 100*size, f = f, ax = axs[1])
