#
# Paper Walkthrough: U-Net
# https://towardsdatascience.com/paper-walkthrough-u-net-98877a2cd33c/
#
# batch normalization is optional
# if not used, to add "bias=True" in Conv2d  #(1), #(2)
#
# Cook your First U-Net in PyTorch
# https://medium.com/data-science/cook-your-first-u-net-in-pytorch-b3297a844cf3
#
# U-Net for brain MRI
# https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/
#

import torch
import torch.nn as nn
from torch.nn.functional import relu


# ---------------------------------------------------------------------------
# UNet
# ---------------------------------------------------------------------------

class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=4, features=64,
                 channels_last=True, classes_last=False):
        super().__init__()

        f1 = features   # 64
        f2 = 2*f1       # 128
        f4 = 2*f2       # 256
        f8 = 2*f4       # 512
        f16 = 2*f8      # 1024

        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input
        # image.
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the
        # exception of the last block which does not include a max-pooling layer.
        # -------
        self.e11 = nn.Conv2d(in_channels, f1, kernel_size=3, padding=1)
        self.e12 = nn.Conv2d(f1, f1, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e21 = nn.Conv2d(f1, f2, kernel_size=3, padding=1)
        self.e22 = nn.Conv2d(f2, f2, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e31 = nn.Conv2d(f2, f4, kernel_size=3, padding=1)
        self.e32 = nn.Conv2d(f4, f4, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e41 = nn.Conv2d(f4, f8, kernel_size=3, padding=1)
        self.e42 = nn.Conv2d(f8, f8, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.e51 = nn.Conv2d(f8, f16, kernel_size=3, padding=1)
        self.e52 = nn.Conv2d(f16, f16, kernel_size=3, padding=1)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(f16, f8, kernel_size=2, stride=2)
        self.d11 = nn.Conv2d(f16, f8, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(f8, f8, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(f8, f4, kernel_size=2, stride=2)
        self.d21 = nn.Conv2d(f8, f4, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(f4, f4, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(f4, f2, kernel_size=2, stride=2)
        self.d31 = nn.Conv2d(f4, f2, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(f2, f2, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(f2, f1, kernel_size=2, stride=2)
        self.d41 = nn.Conv2d(f2, f1, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(f1, f1, kernel_size=3, padding=1)

        # Output layer
        self.outconv = nn.Conv2d(f1, num_classes, kernel_size=1)

        self.channels_last = channels_last
        self.classes_last = classes_last
    # end

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (N, W, H, C) -> (N, C, W, H)
        if self.channels_last:
            x = torch.permute(x, [0, 3, 1, 2])

        # Encoder
        xe11 = relu(self.e11(x))
        xe12 = relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = relu(self.e21(xp1))
        xe22 = relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = relu(self.e31(xp2))
        xe32 = relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = relu(self.e41(xp3))
        xe42 = relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = relu(self.e51(xp4))
        xe52 = relu(self.e52(xe51))

        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = relu(self.d11(xu11))
        xd12 = relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = relu(self.d21(xu22))
        xd22 = relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = relu(self.d31(xu33))
        xd32 = relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = relu(self.d41(xu44))
        xd42 = relu(self.d42(xd41))

        # Output layer [B, C, W, H]
        out = self.outconv(xd42)

        # [B, C, W, H] -> [B, W, H, C]
        if self.classes_last:
            out = torch.permute(out, [0, 2, 3, 1])

        return out
    # end
# end


# ---------------------------------------------------------------------------
# End
# ---------------------------------------------------------------------------
