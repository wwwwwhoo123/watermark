import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, inchannel, ratio=16):
        super(SEBlock, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(inchannel, inchannel // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(inchannel // ratio, inchannel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        y = self.gap(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, with_bn=True, with_relu=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        ]
        if with_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        if with_relu:
            layers.append(nn.ReLU())
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


class ConvTransposeBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.conv(x)


class WatermarkEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1)
        self.act1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=1)
        self.act2 = nn.LeakyReLU(0.2)

        self.down_conv1 = ConvBlock(2, 8, stride=1)
        self.down_conv2 = ConvBlock(8, 16, stride=1)
        self.down_conv3 = ConvBlock(16, 32, stride=1)
        self.down_conv4 = ConvBlock(32, 64, stride=1)
        self.down_conv5 = ConvBlock(64, 128, stride=1)

        self.pool = nn.MaxPool2d(2)
        self.se_down1 = SEBlock(8)
        self.se_down2 = SEBlock(16)
        self.se_down3 = SEBlock(32)
        self.se_down4 = SEBlock(64)
        self.se_down5 = SEBlock(128)
        self.se_up1 = SEBlock(64)
        self.se_up2 = SEBlock(32)
        self.se_up3 = SEBlock(16)
        self.se_up4 = SEBlock(8)

        self.ct_conv1 = ConvTransposeBlock(128, 64)
        self.ct_conv2 = ConvTransposeBlock(64, 32)
        self.ct_conv3 = ConvTransposeBlock(32, 16)
        self.ct_conv4 = ConvTransposeBlock(16, 8)

        self.up_conv1 = ConvBlock(128, 64, stride=1)
        self.up_conv2 = ConvBlock(64, 32, stride=1)
        self.up_conv3 = ConvBlock(32, 16, stride=1)
        self.up_conv4 = ConvBlock(16, 8, stride=1)
        self.up_conv5 = ConvBlock(8, 1, stride=1, with_relu=False)

    def forward(self, x):
        x = self.act1(x)
        x = self.conv1(x)
        # x = self.act1(x)
        x = self.act2(x)
        x = self.conv2(x)
        # x = self.act2(x)

        d1 = self.down_conv1(x)
        p1 = self.se_down1(d1)
        d2 = self.down_conv2(self.pool(p1))
        p2 = self.se_down2(d2)
        d3 = self.down_conv3(self.pool(p2))
        p3 = self.se_down3(d3)
        d4 = self.down_conv4(self.pool(p3))
        p4 = self.se_down4(d4)
        d5 = self.down_conv5(self.pool(p4))
        p5 = self.se_down5(d5)

        u1 = self.ct_conv1(p5)
        c1 = self.up_conv1(torch.cat([u1, p4], dim=1))
        c1 = self.se_up1(c1)
        u2 = self.ct_conv2(c1)
        c2 = self.up_conv2(torch.cat([u2, p3], dim=1))
        c2 = self.se_up2(c2)
        u3 = self.ct_conv3(c2)
        c3 = self.up_conv3(torch.cat([u3, p2], dim=1))
        c3 = self.se_up3(c3)
        u4 = self.ct_conv4(c3)
        c4 = self.up_conv4(torch.cat([u4, p1], dim=1))
        c4 = self.se_up4(c4)

        u5 = self.up_conv5(c4)

        return u5

class WatermarkDecoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.act1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(8, 4, kernel_size=3, stride=2, padding=1)
        self.act2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(4, 1, kernel_size=3, stride=2, padding=1)

        self.down_conv1 = ConvBlock(1, 8, stride=1)
        self.down_conv2 = ConvBlock(8, 16, stride=1)
        self.down_conv3 = ConvBlock(16, 32, stride=1)
        self.down_conv4 = ConvBlock(32, 64, stride=1)
        self.down_conv5 = ConvBlock(64, 128, stride=1)

        self.pool = nn.MaxPool2d(2)
        self.se_down1 = SEBlock(8)
        self.se_down2 = SEBlock(16)
        self.se_down3 = SEBlock(32)
        self.se_down4 = SEBlock(64)
        self.se_down5 = SEBlock(128)
        self.se_up1 = SEBlock(64)
        self.se_up2 = SEBlock(32)
        self.se_up3 = SEBlock(16)
        self.se_up4 = SEBlock(8)

        self.up_conv1 = ConvBlock(128, 64, stride=1)
        self.up_conv2 = ConvBlock(64, 32, stride=1)
        self.up_conv3 = ConvBlock(32, 16, stride=1)
        self.up_conv4 = ConvBlock(16, 8, stride=1)
        self.up_conv5 = ConvBlock(8, 1, stride=1, with_relu=False)

        self.ct_conv1 = ConvTransposeBlock(128, 64)
        self.ct_conv2 = ConvTransposeBlock(64, 32)
        self.ct_conv3 = ConvTransposeBlock(32, 16)
        self.ct_conv4 = ConvTransposeBlock(16, 8)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)

        d1 = self.down_conv1(x)
        p1 = self.se_down1(d1)
        d2 = self.down_conv2(self.pool(p1))
        p2 = self.se_down2(d2)
        d3 = self.down_conv3(self.pool(p2))
        p3 = self.se_down3(d3)
        d4 = self.down_conv4(self.pool(p3))
        p4 = self.se_down4(d4)
        d5 = self.down_conv5(self.pool(p4))
        p5 = self.se_down5(d5)

        u1 = self.ct_conv1(p5)
        c1 = self.up_conv1(torch.cat([u1, p4], dim=1))
        c1 = self.se_up1(c1)
        u2 = self.ct_conv2(c1)
        c2 = self.up_conv2(torch.cat([u2, p3], dim=1))
        c2 = self.se_up2(c2)
        u3 = self.ct_conv3(c2)
        c3 = self.up_conv3(torch.cat([u3, p2], dim=1))
        c3 = self.se_up3(c3)
        u4 = self.ct_conv4(c3)
        c4 = self.up_conv4(torch.cat([u4, p1], dim=1))
        c4 = self.se_up4(c4)

        u5 = self.up_conv5(c4)

        return u5


class WatermarkEncoder256(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=2, padding=1)
        self.act1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(4, 2, kernel_size=3, stride=2, padding=1)
        self.act2 = nn.LeakyReLU(0.2)

        self.down_conv1 = ConvBlock(2, 8, stride=1)
        self.down_conv2 = ConvBlock(8, 16, stride=1)
        self.down_conv3 = ConvBlock(16, 32, stride=1)
        self.down_conv4 = ConvBlock(32, 64, stride=1)
        self.down_conv5 = ConvBlock(64, 128, stride=1)

        self.pool = nn.MaxPool2d(2)
        self.se_down1 = SEBlock(8)
        self.se_down2 = SEBlock(16)
        self.se_down3 = SEBlock(32)
        self.se_down4 = SEBlock(64)
        self.se_down5 = SEBlock(128)
        self.se_up1 = SEBlock(64)
        self.se_up2 = SEBlock(32)
        self.se_up3 = SEBlock(16)
        self.se_up4 = SEBlock(8)

        self.ct_conv1 = ConvTransposeBlock(128, 64)
        self.ct_conv2 = ConvTransposeBlock(64, 32)
        self.ct_conv3 = ConvTransposeBlock(32, 16)
        self.ct_conv4 = ConvTransposeBlock(16, 8)

        self.up_conv1 = ConvBlock(128, 64, stride=1)
        self.up_conv2 = ConvBlock(64, 32, stride=1)
        self.up_conv3 = ConvBlock(32, 16, stride=1)
        self.up_conv4 = ConvBlock(16, 8, stride=1)
        self.up_conv5 = ConvBlock(8, 1, stride=1, with_relu=False)

    def forward(self, x):
        x = self.act1(x)
        x = self.conv1(x)
        # x = self.act1(x)
        x = self.act2(x)
        x = self.conv2(x)
        # x = self.act2(x)

        d1 = self.down_conv1(x)
        p1 = self.se_down1(d1)
        d2 = self.down_conv2(self.pool(p1))
        p2 = self.se_down2(d2)
        d3 = self.down_conv3(self.pool(p2))
        p3 = self.se_down3(d3)
        d4 = self.down_conv4(self.pool(p3))
        p4 = self.se_down4(d4)
        d5 = self.down_conv5(self.pool(p4))
        p5 = self.se_down5(d5)

        u1 = self.ct_conv1(p5)
        c1 = self.up_conv1(torch.cat([u1, p4], dim=1))
        c1 = self.se_up1(c1)
        u2 = self.ct_conv2(c1)
        c2 = self.up_conv2(torch.cat([u2, p3], dim=1))
        c2 = self.se_up2(c2)
        u3 = self.ct_conv3(c2)
        c3 = self.up_conv3(torch.cat([u3, p2], dim=1))
        c3 = self.se_up3(c3)
        u4 = self.ct_conv4(c3)
        c4 = self.up_conv4(torch.cat([u4, p1], dim=1))
        c4 = self.se_up4(c4)

        u5 = self.up_conv5(c4)

        return u5


class WatermarkDecoder256(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.act1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1)
        self.act2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(4, 1, kernel_size=3, stride=2, padding=1)

        self.down_conv1 = ConvBlock(1, 8, stride=1)
        self.down_conv2 = ConvBlock(8, 16, stride=1)
        self.down_conv3 = ConvBlock(16, 32, stride=1)
        self.down_conv4 = ConvBlock(32, 64, stride=1)
        self.down_conv5 = ConvBlock(64, 128, stride=1)

        self.pool = nn.MaxPool2d(2)
        self.se_down1 = SEBlock(8)
        self.se_down2 = SEBlock(16)
        self.se_down3 = SEBlock(32)
        self.se_down4 = SEBlock(64)
        self.se_down5 = SEBlock(128)
        self.se_up1 = SEBlock(64)
        self.se_up2 = SEBlock(32)
        self.se_up3 = SEBlock(16)
        self.se_up4 = SEBlock(8)

        self.up_conv1 = ConvBlock(128, 64, stride=1)
        self.up_conv2 = ConvBlock(64, 32, stride=1)
        self.up_conv3 = ConvBlock(32, 16, stride=1)
        self.up_conv4 = ConvBlock(16, 8, stride=1)
        self.up_conv5 = ConvBlock(8, 1, stride=1, with_relu=False)

        self.ct_conv1 = ConvTransposeBlock(128, 64)
        self.ct_conv2 = ConvTransposeBlock(64, 32)
        self.ct_conv3 = ConvTransposeBlock(32, 16)
        self.ct_conv4 = ConvTransposeBlock(16, 8)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)

        d1 = self.down_conv1(x)
        p1 = self.se_down1(d1)
        d2 = self.down_conv2(self.pool(p1))
        p2 = self.se_down2(d2)
        d3 = self.down_conv3(self.pool(p2))
        p3 = self.se_down3(d3)
        d4 = self.down_conv4(self.pool(p3))
        p4 = self.se_down4(d4)
        d5 = self.down_conv5(self.pool(p4))
        p5 = self.se_down5(d5)

        u1 = self.ct_conv1(p5)
        c1 = self.up_conv1(torch.cat([u1, p4], dim=1))
        c1 = self.se_up1(c1)
        u2 = self.ct_conv2(c1)
        c2 = self.up_conv2(torch.cat([u2, p3], dim=1))
        c2 = self.se_up2(c2)
        u3 = self.ct_conv3(c2)
        c3 = self.up_conv3(torch.cat([u3, p2], dim=1))
        c3 = self.se_up3(c3)
        u4 = self.ct_conv4(c3)
        c4 = self.up_conv4(torch.cat([u4, p1], dim=1))
        c4 = self.se_up4(c4)

        u5 = self.up_conv5(c4)

        return u5

class WatermarkDecoderlatent(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.act1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(8, 4, kernel_size=3, stride=2, padding=1)
        self.act2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(4, 1, kernel_size=3, stride=2, padding=1)

        self.down_conv1 = ConvBlock(1, 8, stride=1)
        self.down_conv2 = ConvBlock(8, 16, stride=1)
        self.down_conv3 = ConvBlock(16, 32, stride=1)
        self.down_conv4 = ConvBlock(32, 64, stride=1)
        self.down_conv5 = ConvBlock(64, 128, stride=1)

        self.pool = nn.MaxPool2d(2)
        self.se_down1 = SEBlock(8)
        self.se_down2 = SEBlock(16)
        self.se_down3 = SEBlock(32)
        self.se_down4 = SEBlock(64)
        self.se_down5 = SEBlock(128)
        self.se_up1 = SEBlock(64)
        self.se_up2 = SEBlock(32)
        self.se_up3 = SEBlock(16)
        self.se_up4 = SEBlock(8)

        self.up_conv1 = ConvBlock(128, 64, stride=1)
        self.up_conv2 = ConvBlock(64, 32, stride=1)
        self.up_conv3 = ConvBlock(32, 16, stride=1)
        self.up_conv4 = ConvBlock(16, 8, stride=1)
        self.up_conv5 = ConvBlock(8, 1, stride=1, with_relu=False)

        self.ct_conv1 = ConvTransposeBlock(128, 64)
        self.ct_conv2 = ConvTransposeBlock(64, 32)
        self.ct_conv3 = ConvTransposeBlock(32, 16)
        self.ct_conv4 = ConvTransposeBlock(16, 8)

    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)

        d1 = self.down_conv1(x)
        p1 = self.se_down1(d1)
        d2 = self.down_conv2(self.pool(p1))
        p2 = self.se_down2(d2)
        d3 = self.down_conv3(self.pool(p2))
        p3 = self.se_down3(d3)
        d4 = self.down_conv4(self.pool(p3))
        p4 = self.se_down4(d4)
        d5 = self.down_conv5(self.pool(p4))
        p5 = self.se_down5(d5)

        u1 = self.ct_conv1(p5)
        c1 = self.up_conv1(torch.cat([u1, p4], dim=1))
        c1 = self.se_up1(c1)
        u2 = self.ct_conv2(c1)
        c2 = self.up_conv2(torch.cat([u2, p3], dim=1))
        c2 = self.se_up2(c2)
        u3 = self.ct_conv3(c2)
        c3 = self.up_conv3(torch.cat([u3, p2], dim=1))
        c3 = self.se_up3(c3)
        u4 = self.ct_conv4(c3)
        c4 = self.up_conv4(torch.cat([u4, p1], dim=1))
        c4 = self.se_up4(c4)

        u5 = self.up_conv5(c4)

        return u5

class WatermarkNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = WatermarkEncoder()
        self.decoder = WatermarkDecoder()

    def forward(self, x, watermarked_image=None):
        compressed_watermark = self.encoder(x)
        if watermarked_image is not None:
            extracted_watermark = self.decoder(watermarked_image)
            return compressed_watermark, extracted_watermark
        return compressed_watermark
