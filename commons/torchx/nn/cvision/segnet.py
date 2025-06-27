#
# https://medium.com/@nikdenof/segnet-from-scratch-using-pytorch-3fe9b4527239
#
import torch
import torch.nn as nn


class ConvReLU(nn.Module):
    def __init__(self, in_c, out_c, kernel_size=3, padding=1) -> None:
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, depth=2, kernel_size=3, padding=1) -> None:
        super(EncoderBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(ConvReLU(in_c if i == 0 else out_c, out_c, kernel_size, padding))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x, ind = self.pool(x)
        return x, ind


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, depth=2, kernel_size=3, padding=1, classification=False) -> None:
        super(DecoderBlock, self).__init__()
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        self.layers = nn.ModuleList()
        for i in range(depth):
            if i == depth - 1 and classification:
                self.layers.append(nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding))
            elif i == depth - 1:
                self.layers.append(ConvReLU(in_c, out_c, kernel_size=kernel_size, padding=padding))
            else:
                self.layers.append(ConvReLU(in_c, in_c, kernel_size=kernel_size, padding=padding))

    def forward(self, x, ind):
        x = self.unpool(x, ind)
        for layer in self.layers:
            x = layer(x)
        return x

# ---------------------------------------------------------------------------
# SegNet
# ---------------------------------------------------------------------------

class SegNet(nn.Module):
    def __init__(self,in_channels=3, num_classes=4, features=64,
                 channels_last=True, classes_last=False) -> None:
        super(SegNet, self).__init__()

        f1 = features
        f2 = 2*f1
        f4 = 2*f2
        f8 = 2*f4
        f16 = 2*f8

        # Encoder
        self.enc0 = EncoderBlock(in_channels, f1)
        self.enc1 = EncoderBlock(f1, f2)
        self.enc2 = EncoderBlock(f2, f4, depth=3)
        self.enc3 = EncoderBlock(f4, f8, depth=3)

        # Bottleneck
        self.bottleneck_enc = EncoderBlock(f8, f16, depth=3)
        self.bottleneck_dec = DecoderBlock(f16, f8, depth=3)

        # Decoder
        self.dec0 = DecoderBlock(f8, f4, depth=3)
        self.dec1 = DecoderBlock(f4, f2, depth=3)
        self.dec2 = DecoderBlock(f2, f1)
        self.dec3 = DecoderBlock(f1, num_classes, classification=True) # No activation

        self.channels_last = channels_last
        self.classes_last = classes_last
    # end

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (N, W, H, C) -> (N, C, W, H)
        if self.channels_last:
            x = torch.permute(x, [0, 3, 1, 2])

        # encoder
        e0, ind0 = self.enc0(x)
        e1, ind1 = self.enc1(e0)
        e2, ind2 = self.enc2(e1)
        e3, ind3 = self.enc3(e2)

        # bottleneck
        b0, indb = self.bottleneck_enc(e3)
        b1 = self.bottleneck_dec(b0, indb)

        # decoder
        d0 = self.dec0(b1, ind3)
        d1 = self.dec1(d0, ind2)
        d2 = self.dec2(d1, ind1)

        # classification layer
        out = self.dec3(d2, ind0)

        # (N, C, W, H) -> (N, W, H, C)
        if self.classes_last:
            out = torch.permute(out, [0, 2, 3, 1])

        return out
    # end
# end

# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
