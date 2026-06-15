import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import init


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, r, drop_rate):
        super(BottleneckBlock, self).__init__()

        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0,
                          stride=drop_rate, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.left = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                      stride=drop_rate, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels // r, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels // r, out_channels=out_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        x = self.left(x)
        scale = self.se(x)
        x = x * scale

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        x = F.relu(x)
        return x


class SENet(nn.Module):
    def __init__(self, in_channels, out_channels, blocks, block_type="BottleneckBlock", r=8, drop_rate=1):
        super(SENet, self).__init__()

        layers = [eval(block_type)(in_channels, out_channels, r, drop_rate)] if blocks != 0 else []
        for _ in range(blocks - 1):
            layer = eval(block_type)(out_channels, out_channels, r, drop_rate)
            layers.append(layer)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class SE_Block(nn.Module):
    def __init__(self, inchannel, ratio=8):
        super(SE_Block, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),  #  c -> c/r
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),  #  c/r -> c
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)


class Adaptor(nn.Module):
    def __init__(self, in_channels, out_channels, initial_value=0.5):
        super(Adaptor, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 2, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels // 2, 3, 2, 1)
        self.conv3 = nn.Conv2d(out_channels // 2, out_channels // 8, 3, 2, 1)
        self.conv4 = nn.Conv2d(out_channels // 8, 1, 3, 2, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels // 2)
        self.bn3 = nn.BatchNorm2d(out_channels // 8)
        self.bn4 = nn.BatchNorm2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.se_block = SE_Block(out_channels)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, latent, watermark):
        latent_w = watermark + latent

        x = self.conv1(latent_w)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.se_block(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        alpha = self.avg_pool(x)

        return alpha


