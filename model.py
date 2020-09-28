import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import defaultdict
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
        return out


class Node(nn.Module):
    def __init__(self, level, node_num, graph, in_channels, out_channels, m, stride=(1, 1, 1), dilation=(1, 1, 1)):
        super(Node, self).__init__()
        self.level = level
        self.edges = graph[node_num]['edges']
        self.edge_total = graph[node_num]['n_totals']
        self.total_edges = sorted(graph[node_num]['edges'] + graph[node_num]['others'])
        # print(self.total_edges)
        self.other_edges = torch.sub(torch.tensor(graph[node_num]['others']), 1).long()
        self.cur_lv = graph[node_num]['level']
        self.channels = [graph[i]['channels'] for i in self.total_edges]
        self.max_channels = max(self.channels)
        self.node_num = node_num

        self.num_edges = len(self.edges)
        self.graph = graph

        self.edge_probs = nn.Parameter(torch.ones([self.edge_total]), requires_grad=True)
        self.switch = nn.Parameter(torch.zeros([self.edge_total]), requires_grad=False)
        self.switch[self.other_edges] = 1.

        self.m = int(m * 2)
        self.layers = nn.ModuleDict()

        self.layers['in'] = nn.ModuleList()
        for node_idx, i in zip(self.total_edges, self.channels):
            lv = graph[node_idx]['level']

            if lv != self.cur_lv and self.cur_lv >= 3:
                interval = abs(lv - self.cur_lv)
                spatial_size = 2 ** (interval - 1)
                self.layers['in'].append(nn.Sequential(
                    ConvBlock(i, out_channels, kernel_size=(1, 1, 1)),
                    ConvBlock(out_channels, out_channels, kernel_size=1, stride=(1, spatial_size, spatial_size))
                ))
            else:
                self.layers['in'].append(ConvBlock(i, out_channels, kernel_size=(1, 1, 1)))

        self.layers['out'] = nn.ModuleList()
        self.layers['out'].append(SpatialConvolution(out_channels, out_channels))
        for _ in range(2, self.m + 1):
            if _ % 2 == 1:
                self.layers['out'].append(SpatialConvolution(out_channels, out_channels))
            else:
                self.layers['out'].append(SpatialTemporalConvolution(out_channels, out_channels))

        if stride == (1, 2, 2):
            self.layers['out'].append(ConvBlock(out_channels, out_channels, kernel_size=1, stride=stride))

        self.layers['out'] = nn.Sequential(*self.layers['out'])

    def forward(self, outs):
        edge_probs = torch.sigmoid(self.edge_probs)
        out = 0.
        for idx, i in enumerate(self.edges):
            out += edge_probs[i - 1] * self.switch[i - 1] * self.layers['in'][i - 1](outs[i])
        return self.layers['out'](out)


class Model(nn.Module):
    def __init__(self, graph, max_level=4, per_n_node=4):
        super(Model, self).__init__()
        self.graph = graph
        self.max_level = max_level
        self.per_n_node = per_n_node

        self.layers = nn.ModuleDict()
        self.outs = defaultdict()

        # self.layers['0'] = nn.ModuleList()
        for _ in range(4):
            self.graph[_ + 1] = {
                'channels': 32,
                'edges': [],
                'm': 1,
                'level': 1,
            }
            if _ in [0, 1]:
                # self.layers['0'][str(_ + 1)] = Stem(3, 32, modality='rgb')
                self.layers[str(_ + 1)] = Stem(3, 32, modality='rgb')
                self.graph[_ + 1]['dilation'] = 4
            else:
                # self.layers['0'][str(_ + 1)] = Stem(3, 32, modality='flow')
                self.layers[str(_ + 1)] = Stem(3, 32, modality='flow')
                self.graph[_ + 1]['dilation'] = 1
        in_channels = 32
        for level in range(1, max_level + 1):
            for n_node in range(1, per_n_node + 1):
                node_num = level * 4 + n_node
                edges = self.graph[node_num]['edges']
                out_channels = self.graph[node_num]['channels']
                dilation = self.graph[node_num]['dilation']
                m = self.graph[node_num]['m']

                if len(edges) == 0:
                    continue

                if level >= 2:
                    stride = (1, 2, 2)
                else:
                    stride = (1, 1, 1)
                node_obj = Node(level, node_num, self.graph, in_channels=in_channels, out_channels=out_channels, m=m, dilation=(dilation, 1, 1), stride=stride)
                self.layers[str(node_num)] = node_obj

        # self.layers = nn.Sequential(self.layers)
        # print(self.layers)
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        # stem
        for _ in range(self.per_n_node):
            # self.outs[0][_ + 1] = self.layers[str(0)][str(_ + 1)](x)
            self.outs[_ + 1] = self.layers[str(_ + 1)](x)

        # layer
        for level in range(1, self.max_level + 1):
            for n_node in range(1, self.per_n_node + 1):
                node_num = level * 4 + n_node
                edges = self.graph[node_num]['edges']

                if len(edges) == 0:
                    continue
                self.outs[node_num] = self.layers[str(node_num)](self.outs)

        out = []
        for _ in range(1, self.per_n_node + 1):
            n_node = self.max_level * 4 + _
            if n_node in self.outs.keys():
                out.append(self.outs[n_node])

        out = torch.cat(out, dim=1)
        channels = out.size(1)
        out = self.pool(out).view(-1, channels)

        return out

    def _evolution(self):
        exclude_total_edges = 0
        current_total_edges = 0
        for key, value in self.graph.items():
            if 'others' in value.keys():
                exclude_total_edges += len(value['others'])
                current_total_edges += len(value['edges'])

        new_total_edges = 0
        for idx, (name, params) in enumerate(self.named_parameters()):
            if 'edge_probs' in name:
                node_num = int(name.split('.')[1])
                exclude_edge = []

                for edge in self.graph[node_num]['edges']:
                    if params[edge - 1] < 0.5:
                        exclude_edge.append(edge - 1)
                        self.graph[node_num]['edges'].remove(edge)
                        self.graph[node_num]['others'].append(edge)
                new_total_edges += len(self.graph[node_num]['edges'])
            if 'switch' in name:
                exclude_edge = torch.tensor(exclude_edge).long()
                params[exclude_edge] = 0.
        edge_ratio = abs(float(current_total_edges - new_total_edges)) / float(exclude_total_edges)

        for idx, (name, params) in enumerate(self.named_parameters()):
            if 'switch' in name:
                node_num = int(name.split('.')[1])
                include_edge_inds = []
                others = self.graph[node_num]['others'].copy()
                for ex_edge in others:
                    if random.random() < edge_ratio:
                        self.graph[node_num]['others'].remove(ex_edge)
                        self.graph[node_num]['edges'].append(ex_edge)
                self.graph[node_num]['others'] = sorted(self.graph[node_num]['others'])
                self.graph[node_num]['edges'] = sorted(self.graph[node_num]['edges'])

                include_edge_inds = torch.tensor(include_edge_inds).long()
                params[include_edge_inds] = 1.


from make_graph import Graph
import pprint

g = Graph()
m = Model(g.graph)
pprint.pprint(m.graph, width=160)
m._evolution()
pprint.pprint(m.graph, width=160)

x = torch.randn([2, 3, 16, 112, 112])
print(m(x).size())
