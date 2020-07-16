import argparse
from pathlib import Path


def load_config():
    parser = argparse.ArgumentParser('AssembleNet Parameters')

    # make graph & model
    parser.add_argument('--dilation', type=tuple, default=(1, 2, 4, 8))
    parser.add_argument('--hidden_size', type=tuple, default=(
                     [2, 4, 8, 16],
                     [8, 16, 32, 64],
                     [8, 16, 32, 64],
                     [8, 16, 32, 64],)
    )
    parser.add_argument('--m', type=tuple, default=(1.5, 2, 3, 1.5))
    parser.add_argument('--max_level', type=int, default=4)
    parser.add_argument('--per_n_nodes', type=int, default=4)

    # reference:
    # https://github.com/kenshohara/3D-ResNets-PyTorch/
    parser.add_argument('--root_path', type=Path, default='E:/')
    parser.add_argument('--video_path', type=Path, default='E:/ucf-101-rawframes')
    parser.add_argument('--annotation_path', type=Path, default='E:/ucf-101-annotation/ucf101_01.json')
    parser.add_argument('--result_path', type=Path, default='E:/ucf-101-results')
    parser.add_argument('--dataset', type=str, default='ucf101')
    parser.add_argument('--n_classes', type=int, default=101)
    parser.add_argument('--sample_size', type=int, default=256)
    parser.add_argument('--sample_duration', type=int, default=32)
    parser.add_argument('--initial_scale', type=float, default=1.0)
    parser.add_argument('--n_scales', type=int, default=5)
    parser.add_argument('--scale_step', type=float, default=0.84089641525)
    parser.add_argument('--train_crop', type=str, default='corner')
    parser.add_argument('--mean_dataset', type=str, default='activitynet')
    parser.add_argument('--value_scale', type=int, default=1)
    parser.add_argument('--input_type', type=str, default='rgb')
    parser.add_argument('--file_type', type=str, default='jpg')
    parser.add_argument(
        '--no_mean_norm',
        action='store_true',
        help='If true, inputs are not normalized by mean.')
    parser.set_defaults(no_mean_norm=False)
    parser.add_argument(
        '--no_std_norm',
        default=True,
        help='If true, inputs are not normalized by standard deviation.')
    parser.add_argument(
        '--no_hflip',
        action='store_true',
        help='If true holizontal flipping is not performed.')
    parser.set_defaults(no_hflip=False)
    parser.add_argument('--colorjitter',
                        default=True,
                        help='If true colorjitter is performed.')
    parser.add_argument('--train_t_crop',
                        default='random',
                        type=str,
                        help=('Temporal cropping method in training. '
                              'random is uniform. '
                              '(random | center)'))
    parser.add_argument(
        '--sample_t_stride',
        default=1,
        type=int,
        help='If larger than 1, input frames are subsampled with the stride.')
    parser.add_argument(
        '--n_val_samples',
        default=2,
        type=int,
        help='Number of validation samples for each activity')

    # optimizer
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--cuda', type=bool, default=True)


    return parser.parse_args()
