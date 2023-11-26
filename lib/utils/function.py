import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import math
import copy
import logging
logging.getLogger(__name__)


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def l1norm(X, dim=-1, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def rbf(x1, x2, gamma):
    """RBF kernel K(x,y) """
    pdist = torch.norm(x1[:, None] - x2, dim=2, p=2)
    return torch.exp(-gamma * pdist)


def vectorized_similarity(inp1, inp2, net):
    inp1 = l2norm(inp1, dim=-1)
    inp2 = l2norm(inp2, dim=-1)
    sim_vec = torch.pow(torch.sub(inp1, inp2), 2)
    sim_vec = l2norm(net(sim_vec), dim=-1)
    return sim_vec


def kl_loss(pred1, pred2, eps=1e-8):
    single_loss = torch.sum(pred2 * torch.log(eps + pred2 / (pred1 + eps)), 1)
    return single_loss


def ce_loss(logit, pred):
    single_loss = torch.sum(-pred * F.log_softmax(logit, dim=-1), dim=-1)
    return single_loss


def diversity_loss(emb, reduction='mean'):
    assert emb.dim() == 3
    assert reduction in ['mean', 'sum']

    nbatch, nhead, ndim = emb.shape
    matrix = emb.bmm(emb.transpose(1, 2))

    I = (torch.eye(nhead) > 0.5).repeat(nbatch, 1, 1)
    I = I.to(emb.device)
    matrix = matrix.masked_fill_(I, 0.0)
    loss = torch.stack([torch.norm(m, p=2) for m in matrix]) / (nhead**2)
    return loss.mean() if reduction == 'mean' else loss.sum()


def mmdrbf_loss(emb1, emb2, gamma=0.5, reduction='mean'):
    assert reduction in ['mean', 'sum']

    ndim = emb1.size(-1)
    emb1 = emb1.view(-1, ndim)
    emb2 = emb2.view(-1, ndim)

    if gamma is None:
        gamma = 1. / ndim
    loss = rbf(emb1, emb1, gamma) - 2 * rbf(emb1, emb2, gamma) + rbf(emb2, emb2, gamma)
    return loss.mean() if reduction == 'mean' else loss.sum()


def sinkhorn_algorithm(dist, bs, n, m, beta=0.5, iteration=None):
    # dist is the distance matrix (bs * n * m)
    if iteration is None:
        iteration = [50, 1]

    sigma = torch.ones(bs, m, 1).to(dist.device) / (m * 1.0)
    T = torch.ones(bs, n, m).to(dist.device)
    A = torch.exp(-dist / beta).float()

    for t in range(iteration[0]):
        Q = A * T  # bs * n * m
        for k in range(iteration[1]):
            delta = 1.0 / (n * torch.bmm(Q, sigma))
            a = torch.bmm(torch.transpose(Q, 1, 2), delta)
            sigma = 1.0 / (float(m) * a)
        T = delta * Q * sigma.transpose(2, 1)

    return T


def drop_instance(emb, dp_rate=0.2):
    assert emb.dim() == 3
    nbatch, ninstance, ndim = emb.size()

    features, lengths = [], []
    rand_list_1 = np.random.rand(nbatch, ninstance)
    rand_list_2 = np.random.rand(nbatch)
    for i in range(nbatch):
        if rand_list_2[i] > dp_rate:
            feat_i = emb[i][np.where(rand_list_1[i] > dp_rate * rand_list_2[i])]
            len_i = len(feat_i)
            pads_i = torch.zeros(ninstance - len_i, ndim).to(emb.device)
            feat_i = torch.cat([feat_i, pads_i], dim=0)
        else:
            feat_i = emb[i]
            len_i = ninstance
        lengths.append(len_i)
        features.append(feat_i)
    features = torch.stack(features, dim=0)
    lengths = torch.tensor(lengths).to(features.device)

    return features, lengths


def position_encoding_1d(dim, len):
    """
    :param dim: dimension of the model
    :param len: length of positions
    :return: length*d_model position matrix
    """
    if dim % 2 != 0:
        raise ValueError("Cannot use sin/cos position encoding with "
                         "odd dim (got dim={:d})".format(dim))
    pe = torch.zeros(len, dim)
    position = torch.arange(0, len).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, dim, 2, dtype=torch.float) *
                          -(math.log(10000.0) / dim)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


def GELU(x):
    """
    GELU(x)=0.5 x (1 + tanh[sqrt(2/π) (x + 0.044715 x^3)])
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2. / math.pi) * (x + 0.044715 * torch.pow(x, 3))))



