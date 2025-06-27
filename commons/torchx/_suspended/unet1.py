import torch
import torch.nn as nn

# ---------------------------------------------------------------------------

class DoubleConvOld(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1, use_batch=True):
        """

        :param in_channels:
        :param out_channels:
        :param use_batch: batch normalization is optional
        """
        super().__init__()

        # in theory it is not necessary to use multiple ReLU layers
        # because the layer has no parameter and then can be reused.
        # However, for clarity, here there are 2 layers

        self.conv_0 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=padding, bias=not use_batch)    #(1)
        self.bn_0 = nn.BatchNorm2d(num_features=out_channels) if use_batch else None
        self.relu_0 = nn.ReLU(inplace=True)

        self.conv_1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=padding, bias=not use_batch)    #(2)
        self.bn_1 = nn.BatchNorm2d(num_features=out_channels) if use_batch else None
        self.relu_1 = self.relu_0 # nn.ReLU(inplace=True)
    # end

    def forward(self, x):
        x = self.conv_0(x)
        x = self.bn_0(x) if self.bn_0 else x
        x = self.relu_0(x)

        x = self.conv_1(x)
        x = self.bn_1(x) if self.bn_1 else x
        x = self.relu_1(x)

        return x
    # end
# end


class DownSampleOld(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = DoubleConv(in_channels=in_channels,  out_channels=out_channels)    #(1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)    #(2)

    def forward(self, x):
        convolved = self.double_conv(x)

        maxpooled = self.maxpool(convolved)

        return convolved, maxpooled    #(3)
    # end
# end


def crop_image(original, expected):    #(1)

    original_w = original.size()[-2]    #(2)
    expected_w = expected.size()[-2]    #(3)
    original_h = original.size()[-1]    #(2)
    expected_h = expected.size()[-1]    #(3)

    difference_w = original_w - expected_w    #(4)
    difference_h = original_h - expected_h    #(4)

    padding_w = difference_w // 2    #(5)
    padding_h = difference_h // 2  # (5)

    cropped = original[:, :, padding_w:original_w-padding_w, padding_h:original_h-padding_h]    #(6)

    return cropped
# end

# ---------------------------------------------------------------------------

class UpSampleOld(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv_transpose = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels,  kernel_size=2, stride=2)    #(1)
        self.double_conv = DoubleConv(
            in_channels=in_channels, out_channels=out_channels)

    def forward(self, x, connection):  # (1)
        x = self.conv_transpose(x)  # (2)

        # cropped_connection = crop_image(connection, x)  # (3)
        # x = torch.cat([x, cropped_connection], dim=1)  # (4)
        x = torch.cat([x, connection], dim=1)

        x = self.double_conv(x)  # (5)

        return x
    # end
# end


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        convolved = self.conv(x)
        maxpooled = self.pool(convolved)
        return convolved, maxpooled
# end

# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, padding = 1):
        super().__init__()
        self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
        self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.c1(x)
        t = self.c2(t)
        return t
# end


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, padding=1):
        super().__init__()
        self.c1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
        self.c2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.c1(x)
        t = self.c2(t)
        return t
# end


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convt = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.uconv = UpConv(in_channels, out_channels)

    def forward(self, x, connection):
        t = self.convt(x)
        t = torch.cat([t, connection], dim=1)
        t = self.uconv(t)
        return t
# end


class UNetOld(nn.Module):
    def __init__(self, in_channels=3, num_classes=4, features=64, channels_last=True):    #(1)
        super().__init__()

        f1 = features
        f2 = 2*f1
        f4 = 2*f2
        f8 = 2*f4
        f16 = 2*f8

        # Encoder    #(2)
        self.downsample_0 = DownSample(in_channels=in_channels, out_channels=f1)
        self.downsample_1 = DownSample(in_channels=f1, out_channels=f2)
        self.downsample_2 = DownSample(in_channels=f2, out_channels=f4)
        self.downsample_3 = DownSample(in_channels=f4, out_channels=f8)

        # Bottleneck    #(3)
        self.bottleneck   = DoubleConv(in_channels=f8, out_channels=f16)

        # Decoder    #(4)
        self.upsample_0   = UpSample(in_channels=f16, out_channels=f8)
        self.upsample_1   = UpSample(in_channels=f8, out_channels=f4)
        self.upsample_2   = UpSample(in_channels=f4, out_channels=f2)
        self.upsample_3   = UpSample(in_channels=f2, out_channels=f1)

        # Output    #(5)
        self.output = nn.Conv2d(in_channels=f1, out_channels=num_classes, kernel_size=1)

        self.channels_last = channels_last
    # end

    def forward(self, x):
        if self.channels_last:
            # (N, W, H, C) -> (N, C, W, H)
            x = torch.permute(x, [0, 3, 1, 2])

        convolved_0, maxpooled_0 = self.downsample_0(x)  # (1)
        convolved_1, maxpooled_1 = self.downsample_1(maxpooled_0)  # (2)
        convolved_2, maxpooled_2 = self.downsample_2(maxpooled_1)  # (3)
        convolved_3, maxpooled_3 = self.downsample_3(maxpooled_2)  # (4)

        x = self.bottleneck(maxpooled_3)

        upsampled_0 = self.upsample_0(x, convolved_3)  # (5)
        upsampled_1 = self.upsample_1(upsampled_0, convolved_2)  # (6)
        upsampled_2 = self.upsample_2(upsampled_1, convolved_1)
        upsampled_3 = self.upsample_3(upsampled_2, convolved_0)

        output = self.output(upsampled_3)

        # if self.channels_last:
        #     output = torch.permute(output, [0, 2, 3, 1])

        # the output must be: [N, C, W, H]
        # where C is the number of classes
        return output
    # end
# end
