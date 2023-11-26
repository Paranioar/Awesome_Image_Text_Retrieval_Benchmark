import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils import clones, l2norm, sinkhorn_algorithm, vectorized_similarity, \
    RegulatorAttnLayer, AttnFiltrationLayer, GraphReasonLayer, GaussianKernel, GaussianKernelGCNLayer

from lib.modules.aggregation import basic_agg

import logging
logger = logging.getLogger(__name__)


def cosine_sim(emb1, emb2, sim_type='mean', sim_mode='pair', sim_smooth=6., **kwargs):
    """Cosine similarity between emb1 and emb2"""
    assert sim_mode in ['pair', 'unpair', 'reduce']
    assert (emb1.dim() <= 4) and (emb2.dim() <= 4)

    emb1 = l2norm(emb1, dim=-1)
    emb2 = l2norm(emb2, dim=-1)

    if emb1.dim() == 2:
        sim = emb1.mm(emb2.t())
    else:
        if sim_mode == 'pair':
            sim = torch.sum(emb1 * emb2, -1)
        elif sim_mode == 'unpair':
            sim = torch.bmm(emb1, emb2.transpose(-2, -1))
        else:
            sim = torch.sum(emb1 * emb2, -1)
            sim = basic_agg(sim, dim=-1, agg_type=sim_type, agg_smooth=sim_smooth)
            sim = sim.unsqueeze(-1)

    return sim


def order_sim(im, s, sim_type='abs', **kwargs):
    """Order embeddings similarity measure $-max(0, s-im)$"""
    assert sim_type in ['raw', 'abs']
    assert (im.dim() == 2) and (s.dim() == 2)

    if sim_type == 'abs':
        im = torch.abs(im)
        s = torch.abs(s)

    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    sim = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return sim


def cosine_dist(emb1, emb2, dist_type='1minus', alpha=0.1, **kwargs):
    """Cosine distance between emb1 and emb2"""
    assert dist_type in ['1minus', 'minus', 'acos', 'threshold']
    assert (emb1.dim() <= 3) and (emb2.dim() <= 3)

    sim = cosine_sim(emb1, emb2, sim_mode='unpair')
    if dist_type == '1minus':
        dist = 1 - sim
    elif dist_type == 'minus':
        dist = - sim
    elif dist_type == 'acos':
        dist = torch.acos(sim)
    else:
        dist = 1 - sim
        min_dist, max_dist = dist.min(), dist.max()
        threshold = min_dist + alpha * (max_dist - min_dist)
        dist = F.relu(dist - threshold)

    return dist.float()


def wd_dist(emb1, emb2, dist_type='threshold', alpha=0.1, beta=0.5, iteration=None, **kwargs):
    """Wasserstein distance between emb1 and emb2"""
    assert (emb1.dim() <= 3) and (emb2.dim() <= 3)
    if emb1.dim() == 2:
        emb1, emb2 = emb1.unsqueeze(0), emb2.unsqueeze(1)
    if iteration is None:
        iteration = [30, 1]

    cos_dist = cosine_dist(emb1, emb2, dist_type, alpha)

    bs, n, m = cos_dist.size()
    T = sinkhorn_algorithm(cos_dist, bs, n, m, beta, iteration)
    temp = torch.bmm(cos_dist.transpose(1, 2), T)

    trace = torch.eye(m).to(emb1.device).unsqueeze(0).repeat(bs, 1, 1)
    dist = (trace * temp).contiguous().view(bs, -1)
    dist = torch.sum(dist, -1)

    return dist


def gwd_dist(emb1, emb2, dist_type='threshold', alpha=0.1, beta=0.1, iteration=None, **kwargs):
    """Gromov Wasserstein distance between emb1 and emb2"""
    assert (emb1.dim() <= 3) and (emb2.dim() <= 3)
    if emb1.dim() == 2:
        emb1, emb2 = emb1.unsqueeze(0), emb2.unsqueeze(1)
    if iteration is None:
        iteration = [5, 20, 1]

    bs, n, m = emb1.size(0), emb1.size(1), emb2.size(1)
    p = (torch.ones(bs, n, 1) / n).to(emb1.device)
    q = (torch.ones(bs, m, 1) / m).to(emb1.device)
    one_n = torch.ones(bs, n, 1).to(emb1.device)
    one_m = torch.ones(bs, m, 1).to(emb1.device)

    cdist_1 = cosine_dist(emb1, emb1, dist_type, alpha)
    cdist_2 = cosine_dist(emb2, emb2, dist_type, alpha)
    cdist_12 = torch.bmm(torch.bmm(cdist_1 ** 2, p), torch.transpose(one_m, 1, 2)) + \
          torch.bmm(one_n, torch.bmm(torch.transpose(q, 1, 2), torch.transpose(cdist_2 ** 2, 1, 2)))
    T = torch.bmm(p, q.transpose(2, 1))

    for i in range(iteration[0]):
        cos_dist = cdist_12 - 2 * torch.bmm(torch.bmm(cdist_1, T), torch.transpose(cdist_2, 1, 2))
        T = sinkhorn_algorithm(cos_dist, bs, n, m, beta, iteration[1:])
    cos_dist = cdist_12 - 2 * torch.bmm(torch.bmm(cdist_1, T), torch.transpose(cdist_2, 1, 2))

    temp = torch.bmm(cos_dist.transpose(1, 2), T.detach())

    trace = torch.eye(m).to(emb1.device).unsqueeze(0).repeat(bs, 1, 1)
    dist = (trace * temp).contiguous().view(bs, -1)
    dist = torch.sum(dist, -1)

    return dist


class SimRAR(nn.Module):
    def __init__(self, emb_dim=1024, sim_dim=256, num_layer=3, sim_func='tanh', *kwargs):
        super(SimRAR, self).__init__()

        self.sim_func = nn.Tanh() if sim_func == 'tanh' else nn.Sigmoid()
        self.sim_tran = nn.Linear(emb_dim, sim_dim)
        self.sim_eval = nn.Sequential(nn.Linear(sim_dim, 1), self.sim_func)

        self.sim_layers = clones(RegulatorAttnLayer(sim_dim), num_layer)

    def forward(self, inp1_l, inp2_l, **kwargs):

        sim_l = vectorized_similarity(inp1_l, inp2_l, self.sim_tran)
        sim_g = l2norm(torch.mean(sim_l, 1), dim=-1)

        for layer in self.sim_layers:
            sim_g = layer(sim_g, sim_l)

        return self.sim_eval(sim_g)


class SimSGRAF(nn.Module):
    def __init__(self, emb_dim=1024, sim_dim=256, sim_type='sgr',
                 num_layer=3, sim_func='sigmoid', bias=True, *kwargs):
        super(SimSGRAF, self).__init__()
        assert sim_type in ['sgr', 'saf']

        self.sim_func = nn.Sigmoid() if sim_func == 'sigmoid' else nn.Tanh()
        self.sim_tran = clones(nn.Linear(emb_dim, sim_dim), 2)
        self.sim_eval = nn.Sequential(nn.Linear(sim_dim, 1), self.sim_func)

        if sim_type == 'sgr':
            self.sim_layers = clones(GraphReasonLayer(sim_dim, bias), num_layer)
        elif sim_type == 'saf':
            self.sim_layers = nn.ModuleList([AttnFiltrationLayer(sim_dim)])

    def forward(self, inp1_l, inp2_l, inp1_g=None, inp2_g=None, *kwargs):

        sim_l = vectorized_similarity(inp1_l, inp2_l, self.sim_tran[0])

        if (inp1_g and inp2_g) is None:
            sim_g = l2norm(torch.mean(sim_l, 1), dim=-1)
        else:
            sim_g = vectorized_similarity(inp1_g, inp2_g, self.sim_tran[1])

        sim_a = torch.cat([sim_g.unsqueeze(1), sim_l], 1)
        for layer in self.sim_layers:
            sim_a = layer(sim_a)

        return self.sim_eval(sim_a[:, 0, :])


class SimGSMN(nn.Module):
    def __init__(self, emb_dim=1024, sim_dim=16, hid_dim=32, sim_type='img',
                 num_layer=8, sim_func='tanh', bias=False, *kwargs):
        super(SimGSMN, self).__init__()
        assert sim_type in ['img', 'txt_dense', 'txt_sparse']

        self.sim_dim = sim_dim
        self.sim_type = sim_type
        self.block_dim = emb_dim // sim_dim

        self.gaussian_kernel = GaussianKernel(sim_type, num_layer)
        self.graph_convolution = GaussianKernelGCNLayer(sim_dim, hid_dim, num_layer, bias)

        self.sim_func = nn.Tanh() if sim_func == 'tanh' else nn.Sigmoid()
        self.out_1 = nn.utils.weight_norm(nn.Linear(hid_dim, hid_dim))
        self.out_2 = nn.utils.weight_norm(nn.Linear(hid_dim, 1))
        self.sim_eval = nn.Sequential(self.out_1, self.sim_func, self.out_2)

    def forward(self, inp1_l, inp2_l, depends=None, *kwargs):
        nbatch, ninstance = inp1_l.size(0), inp1_l.size(1)

        qry_set = torch.split(inp1_l, self.block_dim, dim=2)
        ctx_set = torch.split(inp2_l, self.block_dim, dim=2)

        sim_vec = cosine_sim(torch.stack(qry_set, dim=2),
                             torch.stack(ctx_set, dim=2), sim_mode='pair')

        neighbor_nodes = sim_vec.unsqueeze(2).repeat(1, 1, ninstance, 1)
        neighbor_weights, adjacent_matrix = self.gaussian_kernel(inp1_l, depends)

        if adjacent_matrix is not None:
            neighbor_nodes = neighbor_nodes * adjacent_matrix

        # Propagate matching vector to neighbors to infer phrase correspondence
        hidden_graph = self.graph_convolution(neighbor_nodes, neighbor_weights)
        hidden_graph = hidden_graph.view(nbatch * ninstance, -1)

        # Jointly infer matching score
        sim = self.sim_eval(hidden_graph).view(nbatch, -1).mean(dim=1, keepdim=True)

        return sim


__factory = {
    'SimRAR': SimRAR,
    'SimSGRAF': SimSGRAF,
    'SimGSMN': SimGSMN,
}

__function_factory = {
    'SimCosine': cosine_sim,
    'SimOrder': order_sim,
    'DistCosine': cosine_dist,
    'DistWD': wd_dist,
    'DistGWD': gwd_dist,
}


def init_similarity(name, **kwargs):
    if name in __factory.keys():
        return __factory[name](**kwargs)
    elif name in __function_factory.keys():
        return __function_factory[name]
    else:
        raise KeyError("Unknown similarity module :{}".format(name))


if __name__ == '__main__':
    module = gwd_dist

    TensorA = torch.rand(10, 9, 1024).cuda()
    TensorB = torch.rand(10, 7, 1024).cuda()
    # TensorAg = torch.rand(10, 1024).cuda()
    # TensorBg = torch.rand(10, 1024).cuda()

    dist = module(TensorA, TensorB, iteration=[2, 10, 1])

    print(dist)

