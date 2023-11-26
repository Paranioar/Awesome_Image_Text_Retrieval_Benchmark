"""
# Pytorch implementation for Module Collection of Image-Text Retrieval.
# If you find this code helpful, please cite our relevant publications.
# Writen by Haiwen Diao, 2022
"""

"""Model wrapper"""
import torch
import torch.nn as nn
import torch.nn.init

import numpy as np

from .modules import init_imgencoder, init_capencoder, \
    init_aggregation, init_interaction, init_similarity
from .utils import l2norm, drop_instance

import logging
logger = logging.getLogger(__name__)


class Local2LocalModel(nn.Module):
    def __init__(self, opt, **kwargs):
        super(Local2LocalModel, self).__init__()
        # Build Models
        self.opt = opt
        assert self.opt.attn_mode in ['t2i', 'i2t']

        self.img_enc = init_imgencoder(name=opt.imgenc_name,
                                       img_dim=opt.img_dim,
                                       emb_dim=opt.emb_dim,
                                       head=opt.img_head,
                                       dp_rate=opt.img_dp_rate,
                                       num_layer=opt.img_num_layer,
                                       imgenc_type=opt.imgenc_type,
                                       imgenc_path=opt.imgenc_path,
                                       no_imgnorm=opt.no_imgnorm)
        self.cap_enc = init_capencoder(name=opt.capenc_name,
                                       vocab_size=opt.vocab_size,
                                       word_dim=opt.word_dim,
                                       emb_dim=opt.emb_dim,
                                       num_layer=opt.cap_num_layer,
                                       no_capnorm=opt.no_capnorm)
        self.attn_enc = init_interaction(name=opt.attn_name,
                                         emb_dim=opt.emb_dim,
                                         attn_dim=opt.attn_dim,
                                         attn_type=opt.attn_type,
                                         dp_rate=opt.attn_dp_rate,
                                         attn_smooth=opt.attn_smooth,
                                         num_layer=opt.attn_num_layer)
        self.sim_enc = init_similarity(name=opt.sim_name,
                                       bias=opt.sim_bias,
                                       sim_dim=opt.sim_dim,
                                       hid_dim=opt.hid_dim,
                                       emb_dim=opt.emb_dim,
                                       sim_type=opt.sim_type,
                                       sim_func=opt.sim_func,
                                       num_layer=opt.sim_num_layer)
        self.init_weights()

    def init_weights(self):
        """initialization for the fully connected layer
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward_emb(self, images, captions, cap_lens):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
            cap_lens = cap_lens.cuda()

        # Forward feature encoding
        img_embs = self.img_enc(images)
        cap_embs = self.cap_enc(captions, cap_lens)

        return img_embs, cap_embs, cap_lens

    def forward_sim(self, img_embs, cap_embs, cap_lens):
        """Compute the similarities between images and captions
        """
        sim_all = []
        n_image, n_caption = img_embs.size(0), cap_embs.size(0)

        for i in range(n_caption):
            # get the i-th sentence
            n_word = cap_lens[i]
            cap_i = cap_embs[i, :n_word, :].unsqueeze(0)
            cap_i_expand = cap_i.repeat(n_image, 1, 1)

            query = cap_i_expand if self.opt.attn_mode == 't2i' else img_embs
            context = img_embs if self.opt.attn_mode == 't2i' else cap_i_expand
            query, wcontext, _ = self.attn_enc(query, context)

            sim_i = self.sim_enc(query, wcontext,
                                 # depends=None,
                                 # inp1_g=None, inp2_g=None,
                                 sim_mode='reduce',
                                 sim_type=self.opt.sim_type,
                                 sim_smooth=self.opt.sim_smooth)
            sim_all.append(sim_i)

        # (n_image, n_caption)
        sim_all = torch.cat(sim_all, 1)

        return sim_all

    def forward(self, images, captions, lengths, **kwargs):
        """One training step given images and captions.
        """
        # compute the embeddings
        img_embs, cap_embs, cap_lens = self.forward_emb(images, captions, lengths)
        # compute the similarities
        sims = self.forward_sim(img_embs, cap_embs, cap_lens)

        return sims.permute(1, 0)


class Global2LocalModel(nn.Module):
    def __init__(self, opt, **kwargs):
        super(Global2LocalModel, self).__init__()
        # Build Models
        self.opt = opt
        assert self.opt.attn_mode in ['t2i', 'i2t']

        self.img_enc = init_imgencoder(name=opt.imgenc_name,
                                       img_dim=opt.img_dim,
                                       emb_dim=opt.emb_dim,
                                       head=opt.img_head,
                                       dp_rate=opt.img_dp_rate,
                                       num_layer=opt.img_num_layer,
                                       imgenc_type=opt.imgenc_type,
                                       imgenc_path=opt.imgenc_path,
                                       no_imgnorm=opt.no_imgnorm)
        self.cap_enc = init_capencoder(name=opt.capenc_name,
                                       vocab_size=opt.vocab_size,
                                       word_dim=opt.word_dim,
                                       emb_dim=opt.emb_dim,
                                       num_layer=opt.cap_num_layer,
                                       no_capnorm=opt.no_capnorm)
        self.agg_enc = init_aggregation(name=opt.agg_name,
                                        in_dim=opt.emb_dim,
                                        out_dim=opt.emb_dim,
                                        agg_type=opt.agg_type,
                                        dp_rate=opt.agg_dp_rate,
                                        use_bn=opt.agg_use_bn)
        self.attn_enc = init_interaction(name=opt.attn_name,
                                         emb_dim=opt.emb_dim,
                                         attn_dim=opt.attn_dim,
                                         attn_type=opt.attn_type,
                                         dp_rate=opt.attn_dp_rate,
                                         attn_smooth=opt.attn_smooth,
                                         num_layer=opt.attn_num_layer)
        self.sim_enc = init_similarity(name='SimCosine')
        self.init_weights()

    def init_weights(self):
        """initialization for the fully connected layer
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward_emb(self, images, captions, cap_lens):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
            cap_lens = cap_lens.cuda()

        # Forward feature encoding
        img_embs = self.img_enc(images)
        cap_embs = self.cap_enc(captions, cap_lens)

        img_lens = torch.zeros(img_embs.size(0)).to(img_embs.device)
        img_lens[:] = img_embs.size(1)

        agg_embs = cap_embs if self.opt.attn_mode == 't2i' else img_embs
        agg_lens = cap_lens if self.opt.attn_mode == 't2i' else img_lens

        agg_embs = self.agg_enc(agg_embs, lengths=agg_lens,
                                dim=1, topk=1,
                                # features_agg=None,
                                agg_type=self.opt.agg_type,
                                agg_smooth=self.opt.agg_smooth)
        agg_embs = l2norm(agg_embs, dim=-1)

        if self.opt.attn_mode == 't2i':
            return img_embs, agg_embs, cap_lens
        else:
            return agg_embs, cap_embs, cap_lens

    def forward_sim(self, img_embs, cap_embs, cap_lens):
        """Compute the similarities between images and captions
        """
        n_image, n_caption = img_embs.size(0), cap_embs.size(0)

        if img_embs.dim() == 2:
            sim_all = []
            for i in range(n_caption):
                cap_i = cap_embs[i, :cap_lens[i], :].unsqueeze(0)
                query, wcontext, _ = self.attn_enc(img_embs.unsqueeze(1),
                                                   cap_i.repeat(n_image, 1, 1))

                sim_i = self.sim_enc(query, wcontext, sim_mode='pair')
                sim_all.append(sim_i)

            # (n_image, n_caption)
            sim_all = torch.cat(sim_all, 1)
        else:
            cap_embs = cap_embs.unsqueeze(0).repeat(n_image, 1, 1)
            query, wcontext, _ = self.attn_enc(cap_embs, img_embs)

            sim_all = self.sim_enc(query, wcontext, sim_mode='pair')

        return sim_all

    def forward(self, images, captions, lengths, **kwargs):
        """One training step given images and captions.
        """
        # compute the embeddings
        img_embs, cap_embs, cap_lens = self.forward_emb(images, captions, lengths)
        # compute the similarities
        sims = self.forward_sim(img_embs, cap_embs, cap_lens)

        return sims.permute(1, 0)

    def forward_posemb(self, images, captions, cap_lens):
        """Compute the positive image and caption embeddings
        """
        # Set mini-batch dataset
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
            cap_lens = cap_lens.cuda()

        # Forward feature encoding
        img_embs = self.img_enc(images)
        cap_embs = self.cap_enc(captions, cap_lens)

        img_lens = torch.zeros(img_embs.size(0)).to(img_embs.device)
        img_lens[:] = img_embs.size(1)

        agg_embs = cap_embs if self.opt.attn_mode == 't2i' else img_embs
        agg_lens = cap_lens if self.opt.attn_mode == 't2i' else img_lens

        agg_embs = self.agg_enc(agg_embs, lengths=agg_lens,
                                dim=1, topk=1,
                                # features_agg=None,
                                agg_type=self.opt.agg_type,
                                agg_smooth=self.opt.agg_smooth)
        agg_embs = l2norm(agg_embs, dim=-1)

        att_embs = img_embs if self.opt.attn_mode == 't2i' else cap_embs
        att_lens = img_lens if self.opt.attn_mode == 't2i' else cap_lens

        embs = []
        for i in range(att_embs.size(0)):
            inp_i = att_embs[i, :att_lens[i], :]
            _, out_i, _ = self.attn_enc(agg_embs[i].unsqueeze(0),
                                        inp_i.unsqueeze(0))
            embs.append(out_i.squeeze(0))
        att_embs = torch.cat(embs, 0)

        if self.opt.attn_mode == 't2i':
            return att_embs, agg_embs, cap_lens
        else:
            return agg_embs, att_embs, cap_lens


class Global2GlobalModel(nn.Module):
    def __init__(self, opt, **kwargs):
        super(Global2GlobalModel, self).__init__()
        # Build Models
        self.opt = opt

        self.img_enc = init_imgencoder(name=opt.imgenc_name,
                                       img_dim=opt.img_dim,
                                       emb_dim=opt.emb_dim,
                                       head=opt.img_head,
                                       dp_rate=opt.img_dp_rate,
                                       num_layer=opt.img_num_layer,
                                       imgenc_type=opt.imgenc_type,
                                       imgenc_path=opt.imgenc_path,
                                       no_imgnorm=opt.no_imgnorm)
        self.cap_enc = init_capencoder(name=opt.capenc_name,
                                       vocab_size=opt.vocab_size,
                                       word_dim=opt.word_dim,
                                       emb_dim=opt.emb_dim,
                                       num_layer=opt.cap_num_layer,
                                       no_capnorm=opt.no_capnorm)
        self.agg_img = init_aggregation(name=opt.agg_name,
                                        in_dim=opt.emb_dim,
                                        out_dim=opt.emb_dim,
                                        agg_type=opt.agg_type,
                                        dp_rate=opt.agg_dp_rate,
                                        use_bn=opt.agg_use_bn)
        self.agg_cap = init_aggregation(name=opt.agg_name,
                                        in_dim=opt.emb_dim,
                                        out_dim=opt.emb_dim,
                                        agg_type=opt.agg_type,
                                        dp_rate=opt.agg_dp_rate,
                                        use_bn=opt.agg_use_bn)
        self.sim_enc = init_similarity(name='SimCosine')
        self.init_weights()

    def init_weights(self):
        """initialization for the fully connected layer
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                r = np.sqrt(6.) / np.sqrt(m.in_features + m.out_features)
                m.weight.data.uniform_(-r, r)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward_emb(self, images, captions, cap_lens):
        """Compute the image and caption embeddings
        """
        # Set mini-batch dataset
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()
            cap_lens = cap_lens.cuda()

        # Forward feature encoding
        img_embs = self.img_enc(images)
        cap_embs = self.cap_enc(captions, cap_lens)

        if self.training and (not self.opt.no_imgdrop):
            # Size Augmentation during training, randomly drop grids
            img_embs, img_lens = drop_instance(img_embs, dp_rate=0.2)
        else:
            img_lens = torch.zeros(img_embs.size(0)).to(img_embs.device)
            img_lens[:] = img_embs.size(1)

        img_embs = self.agg_img(img_embs, lengths=img_lens,
                                dim=1, topk=1,
                                # features_agg=None,
                                agg_type=self.opt.agg_type,
                                agg_smooth=self.opt.agg_smooth)

        cap_embs = self.agg_cap(cap_embs, lengths=cap_lens,
                                dim=1, topk=1,
                                # features_agg=None,
                                agg_type=self.opt.agg_type,
                                agg_smooth=self.opt.agg_smooth)

        return img_embs, cap_embs, cap_lens

    def forward_sim(self, img_embs, cap_embs, cap_lens):
        """Compute the similarities between images and captions
        """
        return self.sim_enc(img_embs, cap_embs)

    def forward(self, images, captions, lengths, **kwargs):
        """One training step given images and captions.
        """
        # compute the embeddings
        img_embs, cap_embs, cap_lens = self.forward_emb(images, captions, lengths)
        # compute the similarities
        sims = self.forward_sim(img_embs, cap_embs, cap_lens)

        return sims.permute(1, 0)


__factory = {
    'Local2Local': Local2LocalModel,
    'Global2Local': Global2LocalModel,
    'Global2Global': Global2GlobalModel,
}


def init_model(name, **kwargs):
    if name in __factory.keys():
        return __factory[name](**kwargs)
    else:
        raise KeyError("Unknown image-text model :{}".format(name))
