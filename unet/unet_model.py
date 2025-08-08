import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        if self.use_checkpoint and self.training:
            return checkpoint(self.conv, x)
        else:
            return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, use_checkpoint=False):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, use_checkpoint)
        )

    def forward(self, x):
        return self.pool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, use_checkpoint)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, use_checkpoint)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        if self.use_checkpoint and self.training:
            return checkpoint(self.conv, x, use_reentrant=False)

        else:
            return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, use_checkpoint=False):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64, use_checkpoint)
        self.down1 = Down(64, 128, use_checkpoint)
        self.down2 = Down(128, 256, use_checkpoint)
        self.down3 = Down(256, 512, use_checkpoint)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, use_checkpoint)
        self.up1 = Up(1024, 512 // factor, bilinear, use_checkpoint)
        self.up2 = Up(512, 256 // factor, bilinear, use_checkpoint)
        self.up3 = Up(256, 128 // factor, bilinear, use_checkpoint)
        self.up4 = Up(128, 64, bilinear, use_checkpoint)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def enable_checkpointing(self):
        self.use_checkpoint = True
        for module in self.modules():
            if isinstance(module, (DoubleConv, Up, Down)):
                module.use_checkpoint = True
