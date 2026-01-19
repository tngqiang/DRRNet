import torch
import torch.nn as nn
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, padding_mode="reflect"),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        out = self.conv(x)
        return out


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, in_channels , kernel_size=1, stride=1),
            nn.PixelUnshuffle(2),
            nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, stride=1)
        )
    def forward(self, x):
        return self.seq(x)

class Upsample(nn.Module):
    def __init__(self, channels,  out_channel, upscale_factor=2):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels * (upscale_factor ** 2), kernel_size=1, stride=1),
            nn.PixelShuffle(2),
            nn.Conv2d(channels, out_channel, kernel_size=1, stride=1)
        )

    def forward(self, x):
        return self.seq(x)


class UNet5(nn.Module):

    def __init__(self, input_channel=64):
        super(UNet5, self).__init__()

        # self.down_sample=nn.MaxPool2d(2)
        self.down_sample1=Downsample(64, 64)
        self.down_sample2=Downsample(128, 128)
        self.down_sample3=Downsample(256, 256)

        self.down1 = ConvBlock(input_channel, 64)
        self.down2 = ConvBlock(64, 128)
        self.down3 = ConvBlock(128, 256)
        self.down4 = ConvBlock(256, 512)

        self.unsample4 = Upsample(512, 256)
        self.unsample3 = Upsample(256, 128)
        self.unsample2 = Upsample(128, 64)

        self.up3 = ConvBlock(512,256)
        self.up2 = ConvBlock(256,128)
        self.up1 = ConvBlock(128,64)
        self.last = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True, padding_mode="reflect")
            # nn.Sigmoid()
            # nn.ReLU()
        )

    def forward(self, x):
        # print("x",x.shape)
        d1 = self.down1(x)
        d2 = self.down2(self.down_sample1(d1))
        d3 = self.down3(self.down_sample2(d2))
        d4 = self.down4(self.down_sample3(d3))


        out = self.up3(torch.cat((self.unsample4(d4),d3),1))
        out = self.up2(torch.cat((self.unsample3(out),d2),1))
        out = self.up1(torch.cat((self.unsample2(out),d1),1))
        out = self.last(out)

        return out

from model import TransformerBlock
class TBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(),
            TransformerBlock(out_channels, 8, 4, True, 'LayerNorm'),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    def forward(self, x):
        out = self.conv(x)
        return out

class UNet_L(nn.Module):

    def __init__(self, input_channel=64):
        super(UNet_L, self).__init__()

        # self.down_sample=nn.MaxPool2d(2)
        self.down_sample1=Downsample(64, 64)
        self.down_sample2=Downsample(128, 128)
        self.down_sample3=Downsample(256, 256)

        self.down1 = TBlock(input_channel, 64)
        self.down2 = TBlock(64, 128)
        self.down3 = TBlock(128, 256)
        self.down4 = TBlock(256, 512)

        self.unsample4 = Upsample(512, 256)
        self.unsample3 = Upsample(256, 128)
        self.unsample2 = Upsample(128, 64)

        self.up3 = ConvBlock(512,256)
        self.up2 = ConvBlock(256,128)
        self.up1 = ConvBlock(128,64)
        self.last = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True, padding_mode="reflect")
            # nn.Sigmoid()
            # nn.ReLU()
        )

    def forward(self, x):
        # print("x",x.shape)
        d1 = self.down1(x)
        d2 = self.down2(self.down_sample1(d1))
        d3 = self.down3(self.down_sample2(d2))
        d4 = self.down4(self.down_sample3(d3))


        out = self.up3(torch.cat((self.unsample4(d4),d3),1))
        out = self.up2(torch.cat((self.unsample3(out),d2),1))
        out = self.up1(torch.cat((self.unsample2(out),d1),1))
        out = self.last(out)

        return out


if __name__ == '__main__':
    x = torch.randn(24,64, 128, 128)
    model = UNet5()
    print(model(x).shape)