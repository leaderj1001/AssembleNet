import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Model
from make_graph import Graph


def main():
    g = Graph()
    model = Model(g)


if __name__ == '__main__':
    main()
