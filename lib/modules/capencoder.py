import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from transformers import BertModel

from lib.utils import l2norm

import logging
logger = logging.getLogger(__name__)


# Language Model with BiGRU
class CapBiGRU(nn.Module):
    def __init__(self, vocab_size=0, emb_dim=1024, word_dim=300, num_layers=1, no_capnorm=False, **kwargs):
        super(CapBiGRU, self).__init__()
        self.no_capnorm = no_capnorm

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)
        self.dropout = nn.Dropout(0.4)

        # caption embedding
        self.rnn = nn.GRU(word_dim, emb_dim, num_layers, batch_first=True, bidirectional=True)
        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x, lengths, **kwargs):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x_emb = self.embed(x)
        x_emb = self.dropout(x_emb)

        sorted_lengths, indices = torch.sort(lengths, descending=True)
        x_emb = x_emb[indices]
        inv_ix = indices.clone()
        inv_ix[indices] = torch.arange(0, len(indices)).type_as(inv_ix)

        packed = pack_padded_sequence(x_emb, sorted_lengths.data.tolist(), batch_first=True)
        if torch.cuda.device_count() > 1:
            self.rnn.flatten_parameters()

        # Forward propagate RNN
        out, _ = self.rnn(packed)
        cap_emb, _ = pad_packed_sequence(out, batch_first=True)
        cap_emb = cap_emb[inv_ix]
        cap_emb = (cap_emb[:, :, :int(cap_emb.size(2) // 2)] + cap_emb[:, :, int(cap_emb.size(2) // 2):]) / 2

        # normalization in the joint embedding space
        if not self.no_capnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        # For multi-GPUs
        if cap_emb.size(1) < x_emb.size(1):
            pad_size = x_emb.size(1) - cap_emb.size(1)
            pad_emb = torch.Tensor(cap_emb.size(0), pad_size, cap_emb.size(2))
            if torch.cuda.is_available():
                pad_emb = pad_emb.cuda()
            cap_emb = torch.cat([cap_emb, pad_emb], 1)

        return cap_emb


class CapBERT(nn.Module):
    def __init__(self, emb_dim=1024, word_dim=768, no_capnorm=False, **kwargs):
        super(CapBERT, self).__init__()
        self.no_capnorm = no_capnorm

        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.linear = nn.Linear(word_dim, emb_dim)

    def forward(self, x, lengths, **kwargs):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        bert_attention_mask = (x != 0).float()
        bert_emb = self.bert(x, bert_attention_mask)[0]  # B x N x D
        cap_emb = self.linear(bert_emb)

        # normalization in the joint embedding space
        if not self.no_capnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb


__factory = {
    'CapBiGRU': CapBiGRU,
    'CapBERT': CapBERT,
}


def init_capencoder(name, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown caption encoder :{}".format(name))
    return __factory[name](**kwargs)
