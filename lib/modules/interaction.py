import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils import l2norm, clones, vectorized_similarity, \
    qkv_attention, equal_attention, prob_attention

import logging
logging.getLogger(__name__)


class AttnSCAN(nn.Module):
    def __init__(self, attn_smooth=9.0, **kwargs):
        super(AttnSCAN, self).__init__()
        self.smooth = attn_smooth

    def forward(self, query, context, **kwargs):
        """
        query: (nbatch, nquery, d)
        context: (nbatch, ncontext, d)
        """
        # --> (nbatch, d, nquery)
        queryT = torch.transpose(query, 1, 2)

        # (nbatch, ncontext, d)(nbatch, d, nquery)
        # --> (nbatch, ncontext, nquery)
        attn = torch.bmm(context, queryT)
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
        # --> (nbatch, nquery, ncontext)
        attn = torch.transpose(attn, 1, 2).contiguous()
        # --> (nbatch, nquery, ncontext)
        attn = F.softmax(attn * self.smooth, dim=2)

        # --> (nbatch, ncontext, nquery)
        attnT = torch.transpose(attn, 1, 2).contiguous()
        # --> (nbatch, d, ncontext)
        contextT = torch.transpose(context, 1, 2)

        # --> (nbatch, d, nquery)
        wcontext = torch.bmm(contextT, attnT)
        # --> (nbatch, nquery, d)
        wcontext = torch.transpose(wcontext, 1, 2)

        return query, wcontext, attn


class AttnBFAN(nn.Module):
    def __init__(self, attn_smooth=20.0, attn_type='equal', **kwargs):
        super(AttnBFAN, self).__init__()
        self.smooth = attn_smooth
        self.attn_dict = {'equal': equal_attention, 'prob': prob_attention}
        assert attn_type in self.attn_dict.keys()

        self.attn_func = self.attn_dict[attn_type]

    def forward(self, query, context, **kwargs):
        """
        query: (nbatch, nquery, d)
        context: (nbatch, ncontext, d)
        opt: parameters
        """
        nbatch, nquery, ncontext = context.size(0), query.size(1), context.size(1)

        # Step 1: preassign attention
        # --> (nbatch, d, nquery)
        queryT = torch.transpose(query, 1, 2)

        # (nbatch, ncontext, d)(nbatch, d, nquery)
        attn = torch.bmm(context, queryT)
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
        # --> (nbatch, nquery, ncontext)
        attn = torch.transpose(attn, 1, 2).contiguous()
        # --> (nbatch*nquery, ncontext)
        attn = attn.view(nbatch * nquery, ncontext)
        attn = nn.Softmax(dim=1)(attn * self.smooth)
        # --> (nbatch, nquery, ncontext)
        attn = attn.view(nbatch, nquery, ncontext)

        # Step 2: identify irrelevant fragments
        # Learning an indicator function H, one for relevant, zero for irrelevant
        funcH = self.attn_func(attn, nbatch, nquery, ncontext)

        # Step 3: reassign attention
        tmp_attn = funcH * attn
        attn_sum = torch.sum(tmp_attn, dim=-1, keepdim=True)
        re_attn = tmp_attn / attn_sum

        # --> (nbatch, d, ncontext)
        contextT = torch.transpose(context, 1, 2)
        # --> (nbatch, ncontext, nquery)
        re_attnT = torch.transpose(re_attn, 1, 2).contiguous()

        # --> (nbatch, d, nquery)
        wcontext = torch.bmm(contextT, re_attnT)
        # --> (nbatch, nquery, d)
        wcontext = torch.transpose(wcontext, 1, 2)

        return query, wcontext, re_attn


class AttnCAMP(nn.Module):
    def __init__(self, emb_dim=1024, attn_dim=1024, attn_type='raw', dp_rate=None, **kwargs):
        super(AttnCAMP, self).__init__()
        assert attn_type in ['raw', 'new']
        self.attn_type = attn_type

        self.query_fc = nn.Linear(emb_dim, attn_dim, bias=False)
        self.key_fc = nn.Linear(emb_dim, attn_dim, bias=False)
        self.wcontext_query_fc = nn.Linear(emb_dim, attn_dim, bias=False)
        self.wcontext_key_fc = nn.Linear(emb_dim, attn_dim, bias=False)
        self.out_fc = nn.Sequential(nn.Linear(emb_dim * 2, emb_dim, bias=False),
                                    nn.ReLU(inplace=True))
        self.dropout = nn.Dropout(p=dp_rate) if dp_rate > 0 else None

    def forward(self, query, context, mask=None, **kwargs):

        wcontext, attn = qkv_attention(self.query_fc(query), self.key_fc(context),
                                       context, mask, self.dropout)

        if self.attn_type == 'new':
            wcontext, _ = qkv_attention(self.wcontext_query_fc(wcontext),
                                        self.wcontext_key_fc(wcontext),
                                        wcontext)
            wcontext = l2norm(wcontext, dim=-1)

        gate = F.sigmoid((query * wcontext).sum(dim=-1)).unsqueeze(-1)
        fused = torch.cat((query, wcontext), dim=-1) * gate
        query = self.out_fc(fused) + query
        query = l2norm(query, dim=-1)

        return query, wcontext, attn


class AttnADAPT(nn.Module):

    def __init__(self, emb_dim=1024, attn_smooth=10.0, attn_type='mean', **kwargs):
        super(AttnADAPT, self).__init__()
        self.emb_dim = emb_dim
        self.smooth = attn_smooth
        assert attn_type in ['max', 'mean']
        self.attn_type = attn_type

        self.fc_gamma = nn.Linear(emb_dim, emb_dim)
        self.fc_beta = nn.Linear(emb_dim, emb_dim)
        self.bn = nn.BatchNorm1d(emb_dim, affine=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, context, **kwargs):
        nbatch, nquery, ncontext = context.size(0), query.size(1), context.size(1)
        context = context.permute(0, 2, 1)
        context = self.bn(context).view(nbatch, -1, 1, ncontext)

        gammas = self.fc_gamma(query).permute(0, 2, 1).unsqueeze(-1)
        betas = self.fc_beta(query).permute(0, 2, 1).unsqueeze(-1)

        context = context * (gammas + 1) + betas
        mask = self.softmax(context * self.smooth)
        context = mask * context

        if self.attn_type == 'mean':
            wcontext = context.mean(dim=-1)
        else:
            wcontext = context.max(dim=-1)[0]
        wcontext = wcontext.permute(0, 2, 1)

        return query, wcontext, mask


class AttnIMRAM(nn.Module):

    def __init__(self, emb_dim=1024, attn_smooth=9.0, num_layer=2, **kwargs):
        super(AttnIMRAM, self).__init__()
        self.num_layer = num_layer

        self.linear = nn.Linear(emb_dim * 2, emb_dim)
        self.gate = nn.Linear(emb_dim * 2, emb_dim)
        self.attnscan = AttnSCAN(attn_smooth=attn_smooth)

    def forward(self, query, context, **kwargs):
        query_new = query
        for i in range(self.num_layer):
            _, wcontext, attn = self.attnscan(query_new, context)
            query_new = self.update(query_new, wcontext)
        _, wcontext, attn = self.attnscan(query_new, context)

        return query, wcontext, attn

    def update(self, query, wcontext):

        common = torch.cat([query, wcontext], 2)
        refine = torch.tanh(self.linear(common))
        gate = torch.sigmoid(self.gate(common))

        query = query * gate + refine * (1 - gate)
        query = l2norm(query, dim=-1)

        return query


class AttnRCR(nn.Module):

    def __init__(self, emb_dim=1024, attn_dim=256, attn_smooth=10.0, num_layer=2, **kwargs):
        super(AttnRCR, self).__init__()
        self.emb_dim = emb_dim
        self.num_layer = num_layer
        self.attn_smooth = attn_smooth

        self.common_w = clones(nn.Linear(emb_dim, attn_dim), num_layer)
        self.smooth_w = clones(nn.Sequential(nn.Linear(attn_dim, attn_dim // 2),
                                             nn.Tanh(), nn.Linear(attn_dim // 2, 1)),
                               num_layer)
        self.matrix_w = clones(nn.Sequential(nn.Linear(attn_dim, attn_dim * 2),
                                             nn.Tanh(), nn.Linear(attn_dim * 2, emb_dim)),
                               num_layer)
        self.attnscan = AttnSCAN(attn_smooth=attn_smooth)

    def forward(self, query, context, **kwargs):

        matrix = torch.ones(self.emb_dim).to(query.device)
        smooth = self.attn_smooth

        for i in range(self.num_layer):
            self.attnscan.smooth = smooth
            _, wcontext, attn = self.attnscan(query * matrix, context)
            matrix, smooth = self.update(query, wcontext, matrix, smooth, i)
        self.attnscan.smooth = smooth
        _, wcontext, attn = self.attnscan(query * matrix, context)

        return query, wcontext, attn

    def update(self, query, wcontext, matrix, smooth, index):

        common = vectorized_similarity(query, wcontext, self.common_w[index])
        matrix = (torch.tanh(self.matrix_w[index](common)) + matrix).clamp(min=-1, max=1)
        smooth = torch.relu(self.smooth_w[index](common) + smooth)

        return matrix, smooth


class AttnSAN(nn.Module):

    def __init__(self, emb_dim=1024, dp_rate=0.4, **kwargs):
        super(AttnSAN, self).__init__()

        self.map_query_fc = nn.Linear(emb_dim, emb_dim)
        self.map_context_fc = nn.Linear(emb_dim, emb_dim)
        self.sigmoid = nn.Sigmoid()

        self.query_context_fc = clones(nn.Sequential(nn.Linear(emb_dim, emb_dim),
                                                     nn.Tanh(),
                                                     nn.Dropout(dp_rate)), 2)
        self.common_fc = nn.Linear(emb_dim, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, context, **kwargs):
        queryU, contextU = query.unsqueeze(2), context.unsqueeze(1)
        common = self.sigmoid(self.map_query_fc(queryU) + self.map_context_fc(contextU))

        common = self.query_context_fc[0](common) * self.query_context_fc[1](contextU)
        attn = self.softmax(self.common_fc(common).squeeze(-1))

        wcontext = torch.matmul(attn, context)

        return query, wcontext, attn


__factory = {
    'AttnSCAN': AttnSCAN,
    'AttnBFAN': AttnBFAN,
    'AttnCAMP': AttnCAMP,
    'AttnADAPT': AttnADAPT,
    'AttnIMRAM': AttnIMRAM,
    'AttnRCR': AttnRCR,
    'AttnSAN': AttnSAN,
}


def init_interaction(name, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown interaction module :{}".format(name))
    return __factory[name](**kwargs)


if __name__ == '__main__':
    Attn_module = init_interaction(name='AttnRCR')
    TensorA = torch.rand((5, 2, 1024))
    TensorB = torch.rand((5, 3, 1024))

    _, wTensorB, attn = Attn_module(TensorA, TensorB)
    _, wTensorc, attnc = Attn_module(TensorA, TensorB)
    print(TensorA.shape, TensorB.shape, wTensorB.shape, attn.shape)

