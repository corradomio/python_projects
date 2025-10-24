import torch
import torch.nn as nn
import torchvision
from torch.nn import AdaptiveAvgPool2d


class VGG16(nn.Module):
    def __init__(self, output=10, weights=True, bias=False):
        super().__init__()

        vgg16 = torchvision.models.vgg16(weights=weights)
        self.backbone = vgg16.features[:]
        if weights:
            for layer in self.backbone:
                for p in layer.parameters():
                    p.requires_grad = False

        self.avgpool = AdaptiveAvgPool2d(output_size=(7, 7))

        self.fcl = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=bias),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=bias),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=output, bias=bias)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fcl(x)
        return x
