import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=(1, 1, 1), dilation=(1, 1, 1)):
        super(ConvBlock, self).__init__()
        if isinstance(kernel_size, tuple):
            padding = ((kernel_size[0] + (kernel_size[0] - 1) * (dilation[0] - 1)) // 2, (kernel_size[1] + (kernel_size[1] - 1) * (dilation[1] - 1)) // 2, (kernel_size[2] + (kernel_size[2] - 1) * (dilation[2] - 1)) // 2)
        else:
            padding = (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class SpatialConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, stride=(1, 1, 1)):
        super(SpatialConvolution, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size=1)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size=(1, 3, 3), stride=stride)
        self.conv3 = ConvBlock(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        return out + x


class SpatialTemporalConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=(1, 1, 1), stride=(1, 1, 1)):
        super(SpatialTemporalConvolution, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size=(3, 1, 1), dilation=dilation)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size=(1, 3, 3))
        self.conv3 = ConvBlock(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        return out + x
