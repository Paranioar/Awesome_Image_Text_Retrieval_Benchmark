import torch
import torch.nn as nn
import torch.nn.functional as F

from .function import clones, l1norm, l2norm

import logging
logging.getLogger(__name__)


def sum_attention(layer, query, value, mask=None, dropout=None):
    scores = layer(query).transpose(-2, -1)

    if mask is not None:
        scores = scores.masked_fill_(mask.data.eq(0), -1e9)
    attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)
    return (attn @ value), attn


def qkv_attention(query, key, value, mask=None, dropout=None):
    d_k = key.size(-1)
    scores = (query @ key.transpose(-2, -1)) * (d_k ** -0.5)

    if mask is not None:
        scores = scores.masked_fill_(mask.eq(0), -1e9)
    attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        attn = dropout(attn)

    return (attn @ value), attn


def weight_attention(value, smooth):

    nbatch, ninstance = value.size(0), value.size(1)

    valueT = torch.transpose(value, 1, 2).contiguous()
    attn = torch.bmm(value, valueT)
    attn = attn.view(nbatch * ninstance, ninstance)

    attn = nn.Softmax(dim=1)(attn * smooth)
    attn = attn.view(nbatch, ninstance, ninstance)

    return (attn @ value), attn


def equal_attention(attn, nbatch, nquery, ncontext):
    """
    consider the confidence g(x) for each fragment as equal
    sigma_{j} (xi - xj) = sigma_{j} xi - sigma_{j} xj
    attn: (nbatch, nquery, ncontext)
    """
    funcF = attn * ncontext - torch.sum(attn, dim=-1, keepdim=True)
    attn = torch.where(funcF > 0, torch.ones_like(attn),
                        torch.zeros_like(attn))
    return attn


def prob_attention(attn, nbatch, nquery, ncontext):
    """
    consider the confidence g(x) for each fragment as the sqrt
    of their similarity probability to the query fragment
    sigma_{j} (xi - xj)gj = sigma_{j} xi*gj - sigma_{j} xj*gj
    attn: (nbatch, nquery, ncontext)
    """
    # -> (nbatch, nquery, ncontext, 1)
    xi = attn.unsqueeze(-1).contiguous()
    # -> (nbatch, nquery, 1, ncontext)
    xj = attn.unsqueeze(2).contiguous()
    # -> (nbatch, nquery, 1, ncontext)
    xj_confi = torch.sqrt(xj)

    xi = xi.view(nbatch * nquery, ncontext, 1)
    xj = xj.view(nbatch * nquery, 1, ncontext)
    xj_confi = xj_confi.view(nbatch * nquery, 1, ncontext)

    # -> (nbatch*nquery, ncontext, ncontext)
    term1 = torch.bmm(xi, xj_confi)
    term2 = xj * xj_confi
    # -> (nbatch*nquery, ncontext)
    funcF = torch.sum(term1 - term2, dim=-1)
    funcF = funcF.view(nbatch, nquery, ncontext)

    attn = torch.where(funcF > 0, torch.ones_like(attn),
                        torch.zeros_like(attn))
    return attn


class GatedAttnLayer(nn.Module):
    def __init__(self, dim=1024, head=64, dropout=None):
        super(GatedAttnLayer, self).__init__()
        self.h = head
        self.d_k = dim // head

        self.linears = clones(nn.Linear(dim, dim), 3)
        self.dropout = dropout
        self.bn = nn.BatchNorm1d(dim)

        self.fc_q = nn.Linear(self.d_k, self.d_k)
        self.fc_k = nn.Linear(self.d_k, self.d_k)
        self.fc_g = nn.Linear(self.d_k, self.d_k * 2)

    def forward(self, inp, mask=None):

        nbatch, ninstance, ndim = inp.size()
        query, key, value = [l(x).view(nbatch, ninstance, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (inp, inp, inp))]

        # --> (nbatch, head, ninstance, d_k)
        G = self.fc_q(query) * self.fc_k(key)
        # --> (nbatch, head, ninstance, d_k*2)
        M = F.sigmoid(self.fc_g(G))
        query = query * M[:, :, :, :self.d_k]
        key = key * M[:, :, :, self.d_k:]
        scores = torch.matmul(query, key.transpose(-2, -1)) * (self.d_k ** -0.5)

        if mask is not None:
            scores = scores.masked_fill_(mask.eq(0), -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if self.dropout is not None:
            p_attn = self.dropout(p_attn)

        x = torch.matmul(p_attn, value)
        x = x.transpose(1, 2).contiguous().view(nbatch, ninstance, self.h * self.d_k)
        x = self.bn(x.view(nbatch * ninstance, -1)).view(nbatch, ninstance, -1)

        return x


class ConcatAttnLayer(nn.Module):
    def __init__(self, dim=1024):
        super(ConcatAttnLayer, self).__init__()

        self.dim = dim
        self.linear1 = nn.Linear(dim * 2, dim)
        self.linear2 = nn.Linear(dim, 1, bias=False)

    def forward(self, inp_g, inp_l):
        nbatch, ninstance, _ = inp_l.size()
        inp_g = inp_g.unsqueeze(1).repeat(1, ninstance, 1)
        inputs = torch.cat((inp_l, inp_g), 2).view(-1, self.dim * 2)
        o = self.linear2(F.tanh(self.linear1(inputs)))
        e = o.view(nbatch, ninstance)
        alpha = F.softmax(e, dim=1)
        outputs = torch.bmm(alpha.unsqueeze(1), inp_l).squeeze(1)
        return outputs


class AttnFiltrationLayer(nn.Module):
    def __init__(self, dim):
        super(AttnFiltrationLayer, self).__init__()

        self.attn_fc = nn.Linear(dim, 1)
        self.bn = nn.BatchNorm1d(1)

    def forward(self, inp):
        attn = torch.sigmoid(self.bn(self.attn_fc(inp).permute(0, 2, 1)))
        inp = torch.matmul(l1norm(attn, dim=-1), inp)
        inp = l2norm(inp, dim=-1)
        return inp


class RegulatorAttnLayer(nn.Module):
    def __init__(self, dim):
        super(RegulatorAttnLayer, self).__init__()

        self.fc_q = nn.Sequential(nn.Linear(dim, dim),
                                  nn.Tanh(),
                                  nn.Dropout(0.4))
        self.fc_k = nn.Sequential(nn.Linear(dim, dim),
                                  nn.Tanh(),
                                  nn.Dropout(0.4))
        self.fc_v = nn.Sequential(nn.Linear(dim, 1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inp_g, inp_l):
        glo_q = self.fc_q(inp_g)
        loc_k = self.fc_k(inp_l)
        glo_q = glo_q.unsqueeze(1).repeat(1, loc_k.size(1), 1)

        weights = loc_k.mul(glo_q)
        weights = self.softmax(self.fc_v(weights).squeeze(-1))

        out_g = (weights.unsqueeze(-1) * inp_l).sum(dim=1)
        out_g = l2norm(out_g, dim=-1)

        return out_g


if __name__ == '__main__':
    module = AttnFiltrationLayer(10)
    Tensor = torch.rand(4, 5, 10)
    out = module(Tensor)
    print(out.size())
    print(out[:, 0].size())
