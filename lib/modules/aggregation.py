import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from lib.utils import clones, position_encoding_1d

import logging
logging.getLogger(__name__)

agg_dict = {'lse': torch.sum,
            'max': torch.max, 'min': torch.min,
            'sum': torch.sum, 'mean': torch.mean}


def basic_agg(inp, dim=1, agg_type='mean', agg_smooth=6., **kwargs):
    assert agg_type in ['lse', 'max', 'min', 'sum', 'mean']
    agg_func = agg_dict[agg_type]

    if agg_type in ['sum', 'mean']:
        inp = agg_func(inp, dim=dim)
    elif agg_type in ['max', 'min']:
        inp = agg_func(inp, dim=dim)[0]
    else:
        inp.mul_(agg_smooth).exp_()
        inp = agg_func(inp, dim=dim)
        inp = torch.log(inp) / agg_smooth

    return inp


def variable_agg(inp, lengths=None, dim=1, topk=1, agg_type='max', **kwargs):
    assert agg_type in ['max', 'min']

    out = []
    lengths = lengths.cpu().numpy().tolist()
    for idx, len in enumerate(lengths):
        k = min(topk, len)
        inp_i = inp[idx, :len, :]
        index = inp_i.topk(k, dim=dim-1, largest=(agg_type == 'max'))
        rank_i = inp_i.gather(dim-1, index[1]).mean(dim-1)
        out.append(rank_i)
    out = torch.stack(out, dim=0)

    return out


class AggGRU(nn.Module):
    def __init__(self, in_dim=1024, out_dim=1024, agg_type='f30k', **kwargs):
        super(AggGRU, self).__init__()
        self.agg_type = agg_type

        self.gru = nn.GRU(in_dim, out_dim, 1, batch_first=True)
        if 'f30k' in agg_type:
            self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, features, lengths=None, **kwargs):

        sorted_lengths, indices = torch.sort(lengths, descending=True)
        features = features[indices]
        inv_ix = indices.clone()
        inv_ix[indices] = torch.arange(0, len(indices)).type_as(inv_ix)

        packed = pack_padded_sequence(features, sorted_lengths.data.tolist(), batch_first=True)
        if torch.cuda.device_count() > 1:
            self.gru.flatten_parameters()

        _, hidden = self.gru(packed)
        features = hidden[0]
        features = features[inv_ix]

        if 'f30k' in self.agg_type:
            features = self.bn(features)

        return features


class AggSGRAF(nn.Module):
    def __init__(self, in_dim=1024, out_dim=1024, dp_rate=0.4, use_bn=True, **kwargs):
        super(AggSGRAF, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        bn_layer = nn.BatchNorm1d(out_dim) if use_bn else nn.Identity()
        self.query_context_fc = clones(nn.Sequential(nn.Linear(in_dim, out_dim),
                                                     bn_layer, nn.Tanh(),
                                                     nn.Dropout(dp_rate)), 2)

        self.common_fc = nn.Sequential(nn.Linear(out_dim, 1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, features, lengths=None, features_agg=None, **kwargs):
        nbatch, ninstance, ndim = features.size()

        if features_agg is None:
            features_agg = torch.mean(features, dim=1)

        query = self.query_context_fc[0](features_agg).unsqueeze(1)
        context = self.query_context_fc[1](features.view(nbatch * ninstance, self.in_dim))

        # compute the normalized weights, shape: (nbatch, ninstance)
        common = torch.mul(query, context.view(nbatch, ninstance, self.out_dim))
        weights = self.common_fc(common).squeeze(-1)

        if lengths is not None:
            max_len = int(ninstance)
            mask = torch.arange(max_len).expand(lengths.size(0), max_len).to(lengths.device)
            mask = (mask < lengths.long().unsqueeze(1))
            weights = weights.masked_fill(mask == 0, -1e6)

        # compute final image, shape: (nbatch, ndim)
        weights = self.softmax(weights)
        features = (weights.unsqueeze(2) * features).sum(dim=1)

        return features


class AggGPO(nn.Module):
    def __init__(self, in_dim=32, out_dim=32, **kwargs):
        super(AggGPO, self).__init__()
        self.in_dim = in_dim

        self.pe_database = {}
        self.gru = nn.GRU(in_dim, out_dim, 1, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(out_dim, 1, bias=False)

    def compute_weights(self, lengths):
        max_len = int(lengths.max())
        pe_max_len = self.compute_pe(max_len)
        pes = pe_max_len.unsqueeze(0).repeat(lengths.size(0), 1, 1).to(lengths.device)
        mask = torch.arange(max_len).expand(lengths.size(0), max_len).to(lengths.device)
        mask = (mask < lengths.long().unsqueeze(1)).unsqueeze(-1)
        pes = pes.masked_fill(mask == 0, 0)

        if torch.cuda.device_count() > 1:
            self.gru.flatten_parameters()
        packed = pack_padded_sequence(pes, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out_emb, _ = pad_packed_sequence(out, batch_first=True)
        out_emb = (out_emb[:, :, :out_emb.size(2) // 2] + out_emb[:, :, out_emb.size(2) // 2:]) / 2
        scores = self.fc(out_emb)
        scores[torch.where(mask == 0)] = -10000

        weights = torch.softmax(scores / 0.1, 1)
        return weights, mask

    def forward(self, features, lengths=None, **kwargs):
        """
        :param features: features with shape B x K x D
        :param lengths: B x 1, specify the length of each data sample.
        :return: pooled feature with shape B x D
        """
        pool_weights, mask = self.compute_weights(lengths)

        features = features[:, :int(lengths.max()), :]
        sorted_features = features.masked_fill(mask == 0, -10000)
        sorted_features = sorted_features.sort(dim=1, descending=True)[0]
        sorted_features = sorted_features.masked_fill(mask == 0, 0)

        pooled_features = (sorted_features * pool_weights).sum(1)
        return pooled_features

    def compute_pe(self, length):
        """
        :param length: the length of the sequence
        :return: the positional encoding of the given length
        """
        length = int(length)
        if length in self.pe_database.keys():
            return self.pe_database[length]
        else:
            pe = position_encoding_1d(self.in_dim, length)
            self.pe_database[length] = pe
            return pe


__factory = {
    'AggGRU': AggGRU,
    'AggSGRAF': AggSGRAF,
    'AggGPO': AggGPO,
}

__function_factory = {
    'AggBasic': basic_agg,
    'AggVariable': variable_agg,
}


def init_aggregation(name, **kwargs):
    if name in __factory.keys():
        return __factory[name](**kwargs)
    elif name in __function_factory.keys():
        return __function_factory[name]
    else:
        raise KeyError("Unknown aggregation module :{}".format(name))


if __name__ == '__main__':
    module = AggGRU().cuda()
    TensorA = torch.rand(3, 5, 1024).cuda()
    lens = torch.LongTensor([4, 2, 5]).cuda()

    agg = module(TensorA, lens)
    print('finished')