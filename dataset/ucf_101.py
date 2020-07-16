from pathlib import Path
from dataset.mean import get_mean_std
from dataset.spatial_transforms import *
from dataset.temporal_transforms import *
from dataset.target_transforms import *
from dataset.spatial_transforms import Compose as SpatialCompose
from dataset.temporal_transforms import Compose as TemporalCompose
from dataset.target_transforms import Compose as TargetCompose


def get_normalize_method(mean, std, no_mean_norm, no_std_norm):
    if no_mean_norm:
        if no_std_norm:
            return Normalize([0, 0, 0], [1, 1, 1])
        else:
            return Normalize([0, 0, 0], std)
    else:
        if no_std_norm:
            return Normalize(mean, [1, 1, 1])
        else:
            return Normalize(mean, std)


def ucf101(args, mode='train'):
    if mode == 'train':
        args.mean, args.std = get_mean_std(args.value_scale, args.mean_dataset)

        args.n_input_chanenls = 3
        if args.input_type == 'flow':
            args.n_input_chanenls = 2
            args.mean = args.mean[:2]
            args.std = args.std[:2]

        # spatial_transform
        spatial_transform = []
        if args.train_crop == 'random':
            spatial_transform.append(
                RandomResizedCrop(args.sample_size, (args.train_crop_min_scale, 1.0), (args.train_crop_min_ratio, 1.0 / args.train_crop_min_ratio)))
        elif args.train_crop == 'corner':
            scales = [1.0]
            scale_step = 1 / (2 ** (1 / 4))
            for _ in range(1, 5):
                scales.append(scales[-1] * scale_step)
            spatial_transform.append(MultiScaleCornerCrop(args.sample_size, scales))
        elif args.train_crop == 'center':
            spatial_transform.append(Resize(args.sample_size))
            spatial_transform.append(CenterCrop(args.sample_size))

        normalize = get_normalize_method(args.mean, args.std, args.no_mean_norm, args.no_std_norm)

        if not args.no_hflip:
            spatial_transform.append(RandomHorizontalFlip())
        if args.colorjitter:
            spatial_transform.append(ColorJitter())
        spatial_transform.append(ToTensor())
        if args.input_type == 'flow':
            spatial_transform.append(PickFirstChannels(n=2))
        spatial_transform.append(ScaleValue(args.value_scale))
        spatial_transform.append(normalize)
        spatial_transform = SpatialCompose(spatial_transform)

        assert args.train_t_crop in ['random', 'center']
        temporal_transform = []
        if args.sample_t_stride > 1:
            temporal_transform.append(TemporalSubsampling(args.sample_t_stride))
        if args.train_t_crop == 'random':
            temporal_transform.append(TemporalRandomCrop(args.sample_duration))
        elif args.train_t_crop == 'center':
            temporal_transform.append(TemporalCenterCrop(args.sample_duration))
        temporal_transform = TemporalCompose(temporal_transform)
    elif mode == 'val':
        normalize = get_normalize_method(args.mean, args.std, args.no_mean_norm,
                                         args.no_std_norm)
        spatial_transform = [
            Resize(args.sample_size),
            CenterCrop(args.sample_size),
            ToTensor()
        ]
        if args.input_type == 'flow':
            spatial_transform.append(PickFirstChannels(n=2))
        spatial_transform.extend([ScaleValue(args.value_scale), normalize])
        spatial_transform = Compose(spatial_transform)

        temporal_transform = []
        if args.sample_t_stride > 1:
            temporal_transform.append(TemporalSubsampling(args.sample_t_stride))
        temporal_transform.append(
            TemporalEvenCrop(args.sample_duration, args.n_val_samples))
        temporal_transform = TemporalCompose(temporal_transform)

    return spatial_transform, temporal_transform