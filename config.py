import argparse


def load_config():
    parser = argparse.ArgumentParser('AssembleNet Parameters')

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

    return parser.parse_args()
