import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.backbones import S2VTAttnModel
from lib.utils import l2norm, kl_loss, ce_loss, diversity_loss, mmdrbf_loss

import logging
logging.getLogger(__name__)


class BirankLoss(nn.Module):
    def __init__(self, margin=0, max_violation=False, **kwargs):
        super(BirankLoss, self).__init__()
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, sims, **kwargs):
        # compute image-sentence score matrix
        diagonal = sims.diag().view(sims.size(0), 1)
        d1 = diagonal.expand_as(sims)
        d2 = diagonal.t().expand_as(sims)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + sims - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + sims - d2).clamp(min=0)

        # clear diagonals
        I = torch.eye(sims.size(0)) > .5
        I = I.to(sims.device)
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()


class PolyLoss(nn.Module):
    def __init__(self, margin=0, eps=1e-5, **kwargs):
        super(PolyLoss, self).__init__()
        self.margin = margin
        self.eps = eps

    def forward(self, sims, **kwargs):
        nbatch = sims.size(0)
        simsT = sims.t()
        label = torch.Tensor([i for i in range(nbatch)])

        loss = list()
        for i in range(nbatch):
            pos_pair_ = sims[i][i]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - self.eps]
            neg_pair_ = sims[i][label != label[i]]

            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]
            pos_pair = pos_pair_
            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            pos_loss = torch.clamp(0.2 * torch.pow(pos_pair, 2) - 0.7 * pos_pair + 0.5, min=0)
            neg_pair = max(neg_pair)
            neg_loss = torch.clamp(0.9 * torch.pow(neg_pair, 2) - 0.4 * neg_pair + 0.03, min=0)
            loss.append(pos_loss + neg_loss)

        for i in range(nbatch):
            pos_pair_ = simsT[i][i]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - self.eps]
            neg_pair_ = simsT[i][label != label[i]]

            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]
            pos_pair = pos_pair_
            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            pos_loss = torch.clamp(0.2 * torch.pow(pos_pair, 2) - 0.7 * pos_pair + 0.5, min=0)
            neg_pair = max(neg_pair)
            neg_loss = torch.clamp(0.9 * torch.pow(neg_pair, 2) - 0.4 * neg_pair + 0.03, min=0)
            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True).to(sims.device)

        loss = sum(loss) / nbatch
        return loss


class CMPMLoss(nn.Module):
    def __init__(self, loss_smooth=10.0, eps=1e-8, **kwargs):
        super(CMPMLoss, self).__init__()
        self.smooth = loss_smooth
        self.eps = eps
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sims, labels=None, **kwargs):
        label_mask = labels.view(-1, 1) - labels.view(1, -1)
        label_mask = (torch.abs(label_mask) < 0.5).float()
        label_mask_norm = F.normalize(label_mask, p=1, dim=-1)

        v2t_pred = self.softmax(self.smooth * sims)
        t2v_pred = self.softmax(self.smooth * sims.t())

        v2t_loss = kl_loss(label_mask_norm, v2t_pred, self.eps)
        t2v_loss = kl_loss(label_mask_norm, t2v_pred, self.eps)

        loss = v2t_loss.mean() + t2v_loss.mean()

        return loss


class BCELoss(nn.Module):
    def __init__(self, eps=1e-6, max_violation=False, **kwargs):
        super(BCELoss, self).__init__()
        self.eps = eps
        self.max_violation = max_violation

    def forward(self, sims, **kwargs):
        assert (sims.min()[0] >= 0 and sims.max()[0] <= 1)
        sims = sims.clamp(min=self.eps, max=(1.0 - self.eps))
        de_sims = 1.0 - sims

        label = torch.eye(sims.size(0)).to(sims.device)
        de_label = 1 - label

        sims = torch.log(sims) * label
        de_sims = torch.log(de_sims) * de_label

        if self.max_violation:
            loss = -(sims.sum() + sims.sum() + de_sims.min(1)[0].sum() + de_sims.min(0)[0].sum())
        else:
            loss = -(sims.diag().mean() + de_sims.mean())

        return loss


class NPairLoss(nn.Module):
    def __init__(self, max_violation=False, **kwargs):
        super(NPairLoss, self).__init__()
        self.max_violation = max_violation

    def forward(self, sims, labels=None, img_sims=None, cap_sims=None, **kwargs):
        neg_index = self.get_neg_pairs(labels)
        simsT = sims.t()
        pos_sims = sims.diag().view(sims.size(0), 1)

        if not self.max_violation:
            cost_s = torch.sum(torch.exp(sims[neg_index] - pos_sims), 1)
            cost_im = torch.sum(torch.exp(simsT[neg_index] - pos_sims), 1)
            loss = torch.mean(torch.log(1 + cost_s)) + torch.mean(torch.log(1 + cost_im))
        else:
            cost_s = (sims[neg_index] - pos_sims).max(1)[0]
            cost_im = (simsT[neg_index] - pos_sims).max(1)[0]
            loss = torch.log(1 + cost_s).sum() + torch.log(1 + cost_im).sum()

        return loss

    @staticmethod
    def get_neg_pairs(labels):
        n_pairs = np.arange(len(labels))
        n_negatives = []
        for i in range(len(labels)):
            negative = np.concatenate([n_pairs[:i], n_pairs[i+1:]])
            n_negatives.append(negative)

        n_negatives = np.array(n_negatives)

        return torch.LongTensor(n_negatives)


class AngularLoss(NPairLoss):
    def __init__(self, loss_smooth=1., max_violation=False, **kwargs):
        super(AngularLoss, self).__init__()
        self.smooth = loss_smooth
        self.max_violation = max_violation

    def forward(self, sims, labels=None, img_sims=None, cap_sims=None, **kwargs):
        neg_index = self.get_neg_pairs(labels)
        simsT = sims.t()
        pos_sims = sims.diag().view(sims.size(0), 1)

        cost_s = 4. * self.smooth * (sims[neg_index] + cap_sims[neg_index]) - 2. * (1. + self.smooth) * pos_sims
        cost_im = 4. * self.smooth * (simsT[neg_index] + img_sims[neg_index]) - 2. * (1. + self.smooth) * pos_sims

        if not self.max_violation:
            with torch.no_grad():
                max_s = torch.max(cost_s, dim=1)[0]
                max_im = torch.max(cost_im, dim=1)[0]

            cost_s = torch.exp(cost_s - max_s.unsqueeze(1))
            cost_s = torch.log(torch.exp(-max_s) + torch.sum(cost_s, 1))
            cost_im = torch.exp(cost_im - max_im.unsqueeze(1))
            cost_im = torch.log(torch.exp(-max_im) + torch.sum(cost_im, 1))

            loss = torch.mean(max_s + cost_s) + torch.mean(max_im + cost_im)
        else:
            loss = torch.log(1 + torch.exp(cost_s.max(1)[0])).sum() + \
                   torch.log(1 + torch.exp(cost_im.max(1)[0])).sum()

        return loss


class AOQLoss(nn.Module):
    def __init__(self, marginonline=0.2, marginoffline=0, alpha=0.3, beta=1.5, max_violation=False, **kwargs):
        super(AOQLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.marginonline = marginonline
        self.marginoffline = marginoffline
        self.max_violation = max_violation

    def forward(self, im, s, imhnoff, shnoff, imhnoff2, shnoff2, **kwargs):
        # compute image-sentence score matrix
        scores = im.mm(s.t())
        # disable the positive pair
        scores2 = scores - 10 * torch.eye(len(scores)).cuda()
        # get the score list of the online hard negative pairs
        i2thnon = scores2.max(1)[0]
        t2ihnon = scores2.max(0)[0]

        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.marginonline + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.marginonline + scores - d2).clamp(min=0)

        # clear diagonals
        I = torch.eye(scores.size(0)) > .5
        I = I.to(im.device)
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        # score list of positive pairs (only works for measure: cosine_sim)
        pos = (im * s).sum(1)
        # score list of offline hard negative pairs
        i2thnoff = (im * shnoff).sum(1)
        t2ihnoff = (s * imhnoff).sum(1)
        # beta - (offline - online)/alpha
        cweight = self.beta - (i2thnoff - i2thnon) / self.alpha
        iweight = self.beta - (t2ihnoff - t2ihnon) / self.alpha

        cost_s = cost_s * cweight
        cost_im = cost_im * iweight

        # offline hard negatives without common anchors
        i2thnoff2 = (imhnoff * shnoff).sum(1)
        t2ihnoff2 = (shnoff2 * imhnoff2).sum(1)

        cost_s2 = (self.marginoffline + i2thnoff - pos).clamp(min=0)
        cost_im2 = (self.marginoffline + t2ihnoff - pos).clamp(min=0)
        cost_s3 = (self.marginoffline + i2thnoff2 - pos).clamp(min=0)
        cost_im3 = (self.marginoffline + t2ihnoff2 - pos).clamp(min=0)

        return cost_s.sum() + cost_im.sum() + cost_s2.sum() + cost_im2.sum() + cost_s3.sum() + cost_im3.sum()


class InstanceLoss(nn.Module):
    def __init__(self, emb_dim=1024, hidden_dim=512, num_dataclass=None):
        super(InstanceLoss, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_dataclass)
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, img_embs, cap_embs, labels=None):
        assert (img_embs.dim() == 2 and cap_embs.dim() == 2)

        cost_im = self.loss(self.classifier(img_embs), labels)
        cost_s = self.loss(self.classifier(cap_embs), labels)

        return cost_im + cost_s


class TextGenerationLoss(nn.Module):
    def __init__(self, vocab_size=None, emb_dim=1024):
        super(TextGenerationLoss, self).__init__()
        self.caption_module = S2VTAttnModel(vocab_size=vocab_size, embed_size=emb_dim)
        self.loss = nn.NLLLoss(reduce=False)

    def forward(self, img_embs, cap_labels=None, cap_masks=None):
        seq_probs, _ = self.caption_module(img_embs, cap_labels, 'train')
        loss = self.seq_loss(seq_probs, cap_labels[:, 1:], cap_masks[:, 1:])
        return loss

    def seq_loss(self, logits, target, mask):
        # truncate to the same size
        nbatch = logits.shape[0]
        target = target[:, :logits.shape[1]]
        mask = mask[:, :logits.shape[1]]
        logits = logits.contiguous().view(-1, logits.shape[2])
        target = target.contiguous().view(-1)
        mask = mask.contiguous().view(-1)
        loss = self.loss(logits, target)
        loss = torch.sum(loss * mask) / nbatch
        return loss


__factory = {
    'BirankLoss': BirankLoss,
    'PolyLoss': PolyLoss,
    'CMPMLoss': CMPMLoss,
    'BCELoss': BCELoss,
    'NPairLoss': NPairLoss,
    'AngularLoss': AngularLoss,
}


__function_factory = {
    'DiversityLoss': diversity_loss,
    'MmdRbfLoss': mmdrbf_loss,
}


__plug_into_net_factory = {
    'AOQLoss': AOQLoss,
    'InstanceLoss': InstanceLoss,
    'TextGenerationLoss': TextGenerationLoss
}


def init_lossfunction(name, **kwargs):
    if name in __factory.keys():
        return __factory[name](**kwargs)
    elif name in __function_factory.keys():
        return __function_factory[name]
    else:
        raise KeyError("Unknown loss function :{}".format(name))


if __name__ == '__main__':
    import os
    import torch
    import numpy as np
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    emb = torch.ones((2, 4))
    emb2 = torch.Tensor(np.arange(8)).reshape(2, 4)

