import torch
import torch.utils.data as data

from .processing import collate_fn
from .datasets import RawImageDataset, PrecompRegionDataset

import logging
logger = logging.getLogger(__name__)


def get_loader(data_path, data_name, data_split, cap_tool, opt,
               batch_size=100, shuffle=True, num_workers=2, train=True):

    if opt.imgenc_name in ['ImgCNN']:
        dset = RawImageDataset(data_path, data_name, data_split, cap_tool, opt, train)
    else:
        dset = PrecompRegionDataset(data_path, data_name, data_split, cap_tool, opt, train)

    data_loader = data.DataLoader(dataset=dset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=True,
                                  collate_fn=collate_fn)
    return data_loader


def get_loaders(data_path, data_name, cap_tool, batch_size, workers, opt):
    train_loader = get_loader(data_path, data_name, 'train', cap_tool, opt,
                              batch_size, True, workers, True)
    val_loader = get_loader(data_path, data_name, 'dev', cap_tool, opt,
                            batch_size, False, workers, False)
    return train_loader, val_loader


def get_train_loader(data_path, data_name, cap_tool, batch_size, workers, opt,
                     shuffle=True, train=True):
    train_loader = get_loader(data_path, data_name, 'train', cap_tool, opt,
                              batch_size, shuffle, workers, train)
    return train_loader


def get_val_loader(data_path, data_name, cap_tool, batch_size, workers, opt,
                   shuffle=False, train=False):
    val_loader = get_loader(data_path, data_name, 'dev', cap_tool, opt,
                            batch_size, shuffle, workers, train)
    return val_loader


def get_test_loader(data_path, data_name, data_split, cap_tool, batch_size, workers, opt,
                    shuffle=False, train=False):
    test_loader = get_loader(data_path, data_name, data_split, cap_tool, opt,
                             batch_size, shuffle, workers, train)
    return test_loader