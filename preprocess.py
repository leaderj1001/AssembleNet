from torchvision import get_image_backend
import torch

from pathlib import Path
from dataset.mean import get_mean_std
from dataset.loader import VideoLoader, VideoLoaderFlowHDF5, VideoLoaderHDF5
from dataset.videodataset_multiclips import (VideoDatasetMultiClips, collate_fn)
from dataset.videodataset import VideoDataset
from dataset.ucf_101 import ucf101

import random
import numpy as np


def image_name_formatter(x):
    return f'image_{x:05d}.jpg'


def worker_init_fn(worker_id):
    torch_seed = torch.initial_seed()

    random.seed(torch_seed + worker_id)

    if torch_seed >= 2**32:
        torch_seed = torch_seed % 2**32
    np.random.seed(torch_seed + worker_id)


def load_data(args):
    assert args.dataset in ['kinetics', 'activitynet', 'ucf101', 'hmdb51', 'mit']
    assert args.input_type in ['rgb', 'flow']
    assert args.file_type in ['jpg', 'hdf5']

    if args.file_type == 'jpg':
        assert args.input_type == 'rgb', 'flow input is supported only when input type is hdf5.'

        if get_image_backend() == 'accimage':
            from dataset.loader import ImageLoaderAccImage
            loader = VideoLoader(image_name_formatter, ImageLoaderAccImage())
        else:
            loader = VideoLoader(image_name_formatter)

        video_path_formatter = (lambda root_path, label, video_id: root_path / label / video_id)
    else:
        if args.input_type == 'rgb':
            loader = VideoLoaderHDF5()
        else:
            loader = VideoLoaderFlowHDF5()
        video_path_formatter = (lambda root_path, label, video_id: root_path / label / f'{video_id}.hdf5')

    spatial_transform, temporal_transform = ucf101(args, mode='train')
    train_dataset = VideoDataset(args.video_path,
                                 args.annotation_path,
                                 'training',
                                 spatial_transform=spatial_transform,
                                 temporal_transform=temporal_transform,
                                 target_transform=None,
                                 video_loader=loader,
                                 video_path_formatter=video_path_formatter)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=2,
                                               pin_memory=True,
                                               sampler=None,
                                               worker_init_fn=worker_init_fn)

    if args.file_type == 'jpg':
        assert args.input_type == 'rgb', 'flow input is supported only when input type is hdf5.'

        if get_image_backend() == 'accimage':
            from dataset.loader import ImageLoaderAccImage
            loader = VideoLoader(image_name_formatter, ImageLoaderAccImage())
        else:
            loader = VideoLoader(image_name_formatter)

        video_path_formatter = (
            lambda root_path, label, video_id: root_path / label / video_id)
    else:
        if args.input_type == 'rgb':
            loader = VideoLoaderHDF5()
        else:
            loader = VideoLoaderFlowHDF5()
        video_path_formatter = (lambda root_path, label, video_id: root_path / label / f'{video_id}.hdf5')

    val_data = VideoDataset(
        args.video_path,
        args.annotation_path,
        'validation',
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        # target_transform=target_transform,
        video_loader=loader,
        video_path_formatter=video_path_formatter)

    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=2,
                                             pin_memory=True,
                                             sampler=None,
                                             worker_init_fn=worker_init_fn)
    # assert args.inference_subset in ['train', 'val', 'test']

    if args.file_type == 'jpg':
        assert args.input_type == 'rgb', 'flow input is supported only when input type is hdf5.'

        if get_image_backend() == 'accimage':
            from dataset.loader import ImageLoaderAccImage
            loader = VideoLoader(image_name_formatter, ImageLoaderAccImage())
        else:
            loader = VideoLoader(image_name_formatter)

        video_path_formatter = (
            lambda root_path, label, video_id: root_path / label / video_id)
    else:
        if args.input_type == 'rgb':
            loader = VideoLoaderHDF5()
        else:
            loader = VideoLoaderFlowHDF5()
        video_path_formatter = (lambda root_path, label, video_id: root_path / label / f'{video_id}.hdf5')

    inference_data = VideoDataset(
        args.video_path,
        args.annotation_path,
        'testing',
        spatial_transform=spatial_transform,
        temporal_transform=temporal_transform,
        # target_transform=target_transform,
        video_loader=loader,
        video_path_formatter=video_path_formatter,
        target_type=['video_id', 'segment'])

    inference_loader = torch.utils.data.DataLoader(
        inference_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        worker_init_fn=worker_init_fn)

    return train_loader, val_loader, inference_loader
