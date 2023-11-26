import os
import os.path as osp
import cv2
import json
from imageio import imread

import torch
import torch.utils.data as data
import numpy as np

from .processing import process_caption

import logging
logger = logging.getLogger(__name__)


class RawImageDataset(data.Dataset):
    """
    Load original resource of captions and images for COCO or Flickr30K
    """

    def __init__(self, data_path, data_name, data_split, cap_tool, opt, train):

        self.opt = opt
        self.train = train
        self.data_path = data_path
        self.data_name = data_name
        self.cap_tool = cap_tool
        self.cap_drop = train and (not opt.no_capdrop)
        self.capenc_name = opt.capenc_name
        self.imgenc_type = opt.imgenc_type

        loc_cap = osp.join(data_path, 'precomp')
        loc_image = osp.join(data_path, 'precomp')
        loc_mapping = osp.join(data_path, 'id_mapping.json')
        if 'coco' in data_name:
            self.image_base = osp.join(data_path, 'images')
        else:
            self.image_base = osp.join(data_path, 'flickr30k-images')

        with open(loc_mapping, 'r') as f_mapping:
            self.id_to_path = json.load(f_mapping)

        # Read Captions
        self.captions = []
        with open(osp.join(loc_cap, '%s_caps.txt' % data_split), 'r') as f:
            for line in f:
                self.captions.append(line.strip())

        # Get the image ids
        with open(osp.join(loc_image, '{}_ids.txt'.format(data_split)), 'r') as f:
            image_ids = f.readlines()
            self.images = [int(x.strip()) for x in image_ids]

        # Set related parameters according to the pre-trained backbone **
        self.base_target_size = 256
        self.crop_ratio = 0.875
        self.train_scale_rate = 1
        if hasattr(opt, 'input_scale_factor') and opt.input_scale_factor != 1:
            self.base_target_size = int(self.base_target_size * opt.input_scale_factor)
            logger.info('Input images are scaled by factor {}'.format(opt.input_scale_factor))
        if 'detector' in self.imgenc_type:
            self.pixel_means = np.array([[[102.9801, 115.9465, 122.7717]]])
        else:
            self.imagenet_mean = [0.485, 0.456, 0.406]
            self.imagenet_std = [0.229, 0.224, 0.225]

        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        self.length = len(self.captions)
        self.im_div = 5 if len(self.images) != self.length else 1
        self.id_div = 5

        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000

    def __getitem__(self, index):
        # Convert caption (string) to word ids.
        caption = self.captions[index]
        target = process_caption(self.capenc_name, self.cap_tool, caption, self.cap_drop)

        image_id = self.images[index // self.im_div]
        image_path = os.path.join(self.image_base, self.id_to_path[str(image_id)])
        im_in = np.array(imread(image_path))
        processed_image = self._process_image(im_in)
        image = torch.Tensor(processed_image)
        image = image.permute(2, 0, 1)
        return image, target, index, index // self.id_div

    def __len__(self):
        return self.length

    def _process_image(self, im_in):
        """
        Converts an image into a network input, with pre-processing including re-scaling, padding, etc, and data
        augmentation.
        """
        if len(im_in.shape) == 2:
            im_in = im_in[:, :, np.newaxis]
            im_in = np.concatenate((im_in, im_in, im_in), axis=2)

        if 'detector' in self.imgenc_type:
            im_in = im_in[:, :, ::-1]
        im = im_in.astype(np.float32, copy=True)

        if self.train:
            target_size = self.base_target_size * self.train_scale_rate
        else:
            target_size = self.base_target_size

        # 2. Random crop when in training mode, elsewise just skip
        if self.train:
            crop_ratio = np.random.random() * 0.4 + 0.6
            crop_size_h = int(im.shape[0] * crop_ratio)
            crop_size_w = int(im.shape[1] * crop_ratio)
            processed_im = self._crop(im, crop_size_h, crop_size_w, random=True)
        else:
            processed_im = im

        # 3. Resize to the target resolution
        im_shape = processed_im.shape
        im_scale_x = float(target_size) / im_shape[1]
        im_scale_y = float(target_size) / im_shape[0]
        processed_im = cv2.resize(processed_im, None, None, fx=im_scale_x, fy=im_scale_y,
                                  interpolation=cv2.INTER_LINEAR)

        if self.train:
            if np.random.random() > 0.5:
                processed_im = self._hori_flip(processed_im)

        # Normalization
        if 'detector' in self.imgenc_type:
            processed_im = self._detector_norm(processed_im)
        else:
            processed_im = self._imagenet_norm(processed_im)

        return processed_im

    def _imagenet_norm(self, im_in):
        im_in = im_in.astype(np.float32)
        im_in = im_in / 255
        for i in range(im_in.shape[-1]):
            im_in[:, :, i] = (im_in[:, :, i] - self.imagenet_mean[i]) / self.imagenet_std[i]
        return im_in

    def _detector_norm(self, im_in):
        im_in = im_in.astype(np.float32)
        im_in -= self.pixel_means
        return im_in

    @staticmethod
    def _crop(im, crop_size_h, crop_size_w, random):
        h, w = im.shape[0], im.shape[1]
        if random:
            if w - crop_size_w == 0:
                x_start = 0
            else:
                x_start = np.random.randint(w - crop_size_w, size=1)[0]
            if h - crop_size_h == 0:
                y_start = 0
            else:
                y_start = np.random.randint(h - crop_size_h, size=1)[0]
        else:
            x_start = (w - crop_size_w) // 2
            y_start = (h - crop_size_h) // 2

        cropped_im = im[y_start:y_start + crop_size_h, x_start:x_start + crop_size_w, :]

        return cropped_im

    @staticmethod
    def _hori_flip(im):
        im = np.fliplr(im).copy()
        return im


class PrecompRegionDataset(data.Dataset):
    """
    Load precomputed caption and image features for COCO or Flickr30K
    """

    def __init__(self, data_path, data_name, data_split, cap_tool, opt, train):

        self.opt = opt
        self.train = train
        self.data_path = data_path
        self.data_name = data_name
        self.cap_tool = cap_tool
        self.cap_drop = train and (not opt.no_capdrop)
        self.capenc_name = opt.capenc_name

        loc_cap = osp.join(data_path, 'precomp')
        loc_image = osp.join(data_path, 'precomp')

        # Captions
        self.captions = []
        with open(osp.join(loc_cap, '%s_caps.txt' % data_split), 'r') as f:
            for line in f:
                self.captions.append(line.strip())
        # Image features
        self.images = np.load(os.path.join(loc_image, '%s_ims.npy' % data_split))

        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        self.length = len(self.captions)
        self.im_div = 5 if len(self.images) != self.length else 1
        self.id_div = 5

        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000

    def __getitem__(self, index):
        # Convert caption (string) to word ids
        caption = self.captions[index]
        target = process_caption(self.capenc_name, self.cap_tool, caption, self.cap_drop)

        image = self.images[index // self.im_div]
        image = torch.Tensor(image)
        return image, target, index, index // self.id_div

    def __len__(self):
        return self.length

