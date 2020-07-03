import torch
import torch.nn as nn
import torch.nn.functional as F

import random

from utils import SpatialTemporalConvolution, SpatialConvolution, ConvBlock


class Stem(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=(1, 1, 1), modality='rgb'):
        super(Stem, self).__init__()

        if modality == 'rgb':
            self.conv = nn.Sequential(
                ConvBlock(in_channels, out_channels, kernel_size=(1, 7, 7), stride=(1, 2, 2)),
                ConvBlock(out_channels, out_channels, kernel_size=(5, 1, 1), dilation=dilation),
                nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
            )
        else:
            self.conv = nn.Sequential(
                ConvBlock(in_channels, out_channels, kernel_size=(1, 7, 7), stride=(1, 2, 2)),
                nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
            )

    def forward(self, x):
        out = self.conv(x)
        return out + x


class Node(nn.Module):
    def __init__(self, level, num_edges, in_channels, out_channels, m, stride=(1, 1, 1)):
        super(Node, self).__init__()
        self.level = level
        self.num_edges = num_edges

        self.weights = nn.Parameter(torch.ones([len(self.edges)]), requires_grad=True)

        self.m = int(m * 2)
        self.layers = nn.ModuleList()

        self.layers.append(SpatialConvolution(in_channels, out_channels, stride=stride))
        for _ in range(2, self.m + 1):
            if _ % 2 == 1:
                self.layers.append(SpatialConvolution(out_channels, out_channels))
            else:
                self.layers.append(SpatialTemporalConvolution(out_channels, out_channels))

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


class Model(nn.Module):
    def __init__(self, graph):
        super(Model, self).__init__()
        self.graph = graph
        self.layers = nn.ModuleDict()

        self.layers['0'] = nn.ModuleList()
        for _ in range(4):
            random_dilation = self.dilation[random.randint(0, len(self.dilation) - 1)]
            if _ in [0, 1]:
                self.layers['0'].append(Stem(3, 32, dilation=(random_dilation, 1, 1), modality='rgb'))
            else:
                self.layers['0'].append(Stem(3, 32, modality='flow'))

        init_channels = 32
        for level in range(1, self.level + 1):
            self.layers[str(level)] = nn.ModuleDict()
            for _ in range(4):
                random_out_channels, random_dilation = self.hidden_size[level - 1][random.randint(0, len(self.hidden_size[level - 1]) - 1)], self.dilation[random.randint(0, len(self.dilation) - 1)]

    def forward(self, x):
        pass


m = Model()