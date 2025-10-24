from math import ceil

from torch import nn

basic_mb_params = [
    # k, channels(c), repeats(t), stride(s), kernel_size(k)
    [1, 16, 1, 1, 3],
    [6, 24, 2, 2, 3],
    [6, 40, 2, 2, 5],
    [6, 80, 3, 2, 3],
    [6, 112, 3, 1, 5],
    [6, 192, 4, 2, 5],
    [6, 320, 1, 1, 3],
]

alpha, beta = 1.2, 1.1

scale_values = {
    # (phi, resolution, dropout)
    "b0": (0, 224, 0.2),
    "b1": (0.5, 240, 0.2),
    "b2": (1, 260, 0.3),
    "b3": (2, 300, 0.3),
    "b4": (3, 380, 0.4),
    "b5": (4, 456, 0.4),
    "b6": (5, 528, 0.5),
    "b7": (6, 600, 0.5),
}


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, groups=1):
        super(ConvBlock, self).__init__()
        self.cnnblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size,
                      stride, padding, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.SiLU())

    def forward(self, x):
        return self.cnnblock(x)


class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduced_dim):
        super(SqueezeExcitation, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # C x H x W -> C x 1 x 1
            nn.Conv2d(in_channels, reduced_dim, 1),
            nn.SiLU(),
            nn.Conv2d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)


class MBBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, padding, ratio, reduction=2,
                 ):
        super(MBBlock, self).__init__()
        # self.use_residual = in_channels == out_channels and stride == 1
        hidden_dim = in_channels * ratio
        self.expand = in_channels != hidden_dim

        # This is for squeeze and excitation block
        reduced_dim = int(in_channels / reduction)

        if self.expand:
            self.expand_conv = ConvBlock(in_channels, hidden_dim,
                                         kernel_size=3, stride=1, padding=1)

        self.conv = nn.Sequential(
            ConvBlock(hidden_dim, hidden_dim, kernel_size,
                      stride, padding, groups=hidden_dim),
            SqueezeExcitation(hidden_dim, reduced_dim),
            nn.Conv2d(hidden_dim, out_channels, 1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, inputs):
        if self.expand:
            x = self.expand_conv(inputs)
        else:
            x = inputs
        return self.conv(x)


class EfficientNet(nn.Module):
    def __init__(self, model_name: str, output: int):
        super(EfficientNet, self).__init__()
        phi, resolution, dropout = scale_values[model_name]
        self.dropout = dropout
        self.depth_factor, self.width_factor = alpha ** phi, beta ** phi
        self.last_channels: int = ceil(1280 * self.width_factor)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.feature_extractor()
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.last_channels, output),
        )
        self.in_shape = (3, resolution, resolution)

    def feature_extractor(self):
        channels = int(32 * self.width_factor)
        features: list[nn.Module] = [ConvBlock(3, channels, 3, stride=2, padding=1)]
        in_channels = channels

        for k, c_o, repeat, s, n in basic_mb_params:
            # For numeric stability, we multiply and divide by 4
            out_channels = 4 * ceil(int(c_o * self.width_factor) / 4)
            num_layers = ceil(repeat * self.depth_factor)

            for layer in range(num_layers):
                if layer == 0:
                    stride = s
                else:
                    stride = 1
                features.append(
                    # in_channels, out_channels, kernel_size, stride, padding, ratio, reduction=2,
                    MBBlock(in_channels, out_channels,
                            # expand_ratio=k,
                            ratio=k,
                            stride=stride,
                            kernel_size=n,
                            padding=n // 2
                            )
                )
                in_channels = out_channels

        features.append(
            ConvBlock(in_channels, self.last_channels,
                      kernel_size=1, stride=1, padding=0)
        )
        self.extractor = nn.Sequential(*features)

    def forward(self, x):
        assert x.shape[-3:] == self.in_shape, f"Expected input shape {self.in_shape}, got {x.shape}"
        x = self.avgpool(self.extractor(x))
        return self.classifier(self.flatten(x))


class EfficientNetV2(EfficientNet):
    def __init__(self, model_name: str, output: int):
        super().__init__(model_name, output)
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout),

            nn.Linear(self.last_channels, output),

            # nn.Linear(self.last_channels, 512),
            # nn.ReLU(),
            # nn.Linear(512, output)

            # nn.Linear(self.last_channels, 1024),
            # nn.ReLU(),
            # nn.Linear(1024, 256),
            # nn.ReLU(),
            # nn.Linear(256, output)

            # nn.Linear(self.last_channels, 2048),
            # nn.ReLU(),
            # nn.Linear(2048, 1024),
            # nn.ReLU(),
            # nn.Linear(1024, 256),
            # nn.ReLU(),
            # nn.Linear(256, output)
        )
