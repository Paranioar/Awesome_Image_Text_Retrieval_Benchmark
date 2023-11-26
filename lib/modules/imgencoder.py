import torch
import torch.nn as nn

from lib.backbones import MLP, ResnetFeatureExtractor
from lib.utils import l2norm, clones, GatedAttnLayer, RsGCNLayer

import logging
logger = logging.getLogger(__name__)


class ImgFC(nn.Module):
    def __init__(self, img_dim=2048, emb_dim=1024, no_imgnorm=False, **kwargs):
        super(ImgFC, self).__init__()
        self.no_imgnorm = no_imgnorm

        self.fc = nn.Linear(img_dim, emb_dim)

    def forward(self, images, **kwargs):
        """Extract image feature vectors."""
        img_emb = self.fc(images)

        if not self.no_imgnorm:
            img_emb = l2norm(img_emb, dim=-1)

        return img_emb


class ImgMLP(nn.Module):
    def __init__(self, img_dim=2048, emb_dim=1024, no_imgnorm=False, **kwargs):
        super(ImgMLP, self).__init__()
        self.no_imgnorm = no_imgnorm

        self.fc = nn.Linear(img_dim, emb_dim)
        self.mlp = MLP(img_dim, emb_dim // 2, emb_dim, 2)

    def forward(self, images, **kwargs):
        """Extract image feature vectors."""
        img_emb = self.fc(images)
        img_emb = self.mlp(images) + img_emb

        if not self.no_imgnorm:
            img_emb = l2norm(img_emb, dim=-1)

        return img_emb


class ImgCNN(nn.Module):
    def __init__(self, img_dim=2048, emb_dim=1024, imgenc_type=None, imgenc_path=None, no_imgnorm=False, **kwargs):
        super(ImgCNN, self).__init__()

        self.backbone = ResnetFeatureExtractor(imgenc_type, imgenc_path, fixed_blocks=2)
        self.image_encoder = init_imgencoder(name='ImgMLP', img_dim=img_dim,
                                             emb_dim=emb_dim, no_imgnorm=no_imgnorm)
        self.backbone_freezed = False

    def forward(self, images, **kwargs):
        """Extract image feature vectors."""
        img_emb = self.backbone(images)
        img_emb = self.image_encoder(img_emb)

        return img_emb

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info('Backbone freezed.')

    def unfreeze_backbone(self, fixed_blocks):
        for param in self.backbone.parameters():  # open up all params first, then adjust the base parameters
            param.requires_grad = True
        self.backbone.set_fixed_blocks(fixed_blocks)
        self.backbone.unfreeze_base()
        logger.info('Backbone unfreezed, fixed blocks {}'.format(self.backbone.get_fixed_blocks()))


class ImgVSRN(nn.Module):

    def __init__(self, img_dim=2048, emb_dim=1024, num_layer=4, imgenc_type='f30k', no_imgnorm=False, **kwargs):
        super(ImgVSRN, self).__init__()
        self.emb_dim = emb_dim
        self.imgenc_type = imgenc_type
        self.no_imgnorm = no_imgnorm

        # GCN reasoning
        self.fc = nn.Linear(img_dim, emb_dim)
        self.GCNLayers = clones(RsGCNLayer(emb_dim, emb_dim), num_layer)

    def forward(self, images, **kwargs):
        """Extract image feature vectors."""
        img_emb = self.fc(images)

        if 'f30k' not in self.imgenc_type:
            img_emb = l2norm(img_emb, dim=1)

        # GCN reasoning
        # -> B,D,N
        img_emb = img_emb.permute(0, 2, 1)
        for GCNLayer in self.GCNLayers:
            img_emb = GCNLayer(img_emb)
        # -> B,N,D
        img_emb = img_emb.permute(0, 2, 1)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            img_emb = l2norm(img_emb, dim=1)

        return img_emb


class ImgCAMERA(nn.Module):
    def __init__(self, img_dim=2048, emb_dim=1024, num_layer=1, head=64, dp_rate=None, no_imgnorm=False, **kwargs):
        super(ImgCAMERA, self).__init__()
        self.emb_dim = emb_dim
        self.no_imgnorm = no_imgnorm

        self.fc = nn.Linear(img_dim, emb_dim)
        dropout = nn.Dropout(p=dp_rate) if dp_rate > 0 else None
        self.GALayers = clones(GatedAttnLayer(emb_dim, head, dropout), num_layer)

    def forward(self, images, **kwargs):
        """Extract image feature vectors."""
        img_emb = self.fc(images)
        img_emb = l2norm(img_emb, dim=1)

        for GALayer in self.GALayers:
            img_emb += GALayer(img_emb, None)

        # normalize in the joint embedding space
        if not self.no_imgnorm:
            img_emb = l2norm(img_emb, dim=1)

        return img_emb


__factory = {
    'ImgFC': ImgFC,
    'ImgMLP': ImgMLP,
    'ImgCNN': ImgCNN,
    'ImgVSRN': ImgVSRN,
    'ImgCAMERA': ImgCAMERA,
}


def init_imgencoder(name, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown image encoder :{}".format(name))
    return __factory[name](**kwargs)


if __name__ == '__main__':
    model = init_imgencoder(name='ImgCNN', imgenc_type='detector',
                            imgenc_path='/home/dhw/DHW_workspace/project/Cross-Self/data/weights/original_updown_backbone.pth')
    model.cuda()
    model = nn.DataParallel(model)

    module_list = list()
    for name, param in model.module.backbone.named_parameters():
        module_list.append('backbone.'+name)
    print(module_list)

    rest_module_list = list()
    for name, param in model.module.named_parameters():
        if name not in module_list:
            rest_module_list.append(name)
    print(rest_module_list)