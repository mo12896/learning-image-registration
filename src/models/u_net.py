
import torch
import torch.nn as nn
import torch.nn.functional as F



class ConvBlock(nn.Module):
    """

    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv0 = nn.Conv2D(in_channels, out_channels, kernel_size=3)
        self.conv1 = nn.Conv2D(out_channels, out_channels, kernel_size=3)
        self.bn0 = nn.BatchNorm2d(out_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu0 = nn.ReLU()
        self.relu1 = nn.ReLU()

    def forward(self, inputs):
        x = self.conv0(inputs)
        x = self.relu0(x)
        x = self.bn0(x)

        x = self.conv1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        return x

class EncoderBlock(nn.Module):
    """

    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = EncoderBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2, stride=2)

        def forward(self, inputs):
            before_pooling = self.conv(inputs)
            p = self.pool(before_pooling)
            return before_pooling, p


class DecoderBlock(nn.Module):
    """

    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv = ConvBlock(out_channels*2, out_channels)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = self.relu(x)
        x = self.bn(x)
        x = torch.cat((x, skip), axis=1)
        x = self.conv(x)
        return x


class UNetEncoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = EncoderBlock(self.in_channels, 64)
        self.conv2 = EncoderBlock(64, 128)
        self.conv3 = EncoderBlock(128, 256)
        self.conv4 = EncoderBlock(256, 512)

        def forward(self, inputs):
            e1, p1 = self.conv1(inputs)
            e2, p2 = self.conv2(p1)
            e3, p3 = self.conv3(p2)
            e4, p4 = self.conv4(p3)
            return p4, (e1, e2, e3, e4)


class UNetDecoder(nn.Module):
    """

    """
    def __init__(self, classes):
        super().__init__()

        self.up1 = DecoderBlock(1024, 512)
        self.up2 = DecoderBlock(512, 256)
        self.up3 = DecoderBlock(256, 128)
        self.up4 = DecoderBlock(128, 64)

    def forward(self, inputs, encodings):
        e1, e2, e3, e4 = encodings
        d1 = self.up1(inputs, e4)
        d2 = self.up2(d1, e3)
        d3 = self.up3(d2, e2)
        d4 = self.up4(d3, e1)
        return d4


class UNet(nn.Module):
    """

    """
    def __init__(self, in_channels, classes):
        super().__init__()
        self.in_channels = in_channels
        self.classes = classes

        self.encoder = UNetEncoder(in_channels)
        self.bn = ConvBlock(512, 1024)
        self.decoder = UNetDecoder()
        self.outputs = nn.Conv2d(64, classes, kernel_size=1)

    def forward(self, inputs):
        encoder_out, encodings = self.encoder(inputs)
        bottle_neck = self.bn(encoder_out)
        decoder_out = self.decoder(bottle_neck, encodings)
        outputs = self.outputs(decoder_out)
        return outputs

