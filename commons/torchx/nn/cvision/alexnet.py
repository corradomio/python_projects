#
# https://www.digitalocean.com/community/tutorials/alexnet-pytorch
#

import torch.nn as nn


class AlexNet(nn.Module):
    def __init__(self, output=10):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            # nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        data_len = 256 * 6 * 6
        # data_len = 6400
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(data_len, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        # self.fc2 = nn.Sequential(
        #     nn.Linear(4096, num_classes)
        # )
        self.fc2 = nn.Sequential(
            nn.Linear(4096, output))

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.avgpool(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

class AlexNetV2(AlexNet):
    def __init__(self, output=10):
        super(AlexNetV2, self).__init__(output=output)
        self.fc2 = self.fc2 = nn.Sequential(
            # nn.Linear(4096, output)

            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, output)

            # nn.Linear(4096, 1024),
            # nn.ReLU(),
            # nn.Linear(1024, 256),
            # nn.ReLU(),
            # nn.Linear(256, output)

            # nn.Linear(4096, 2048),
            # nn.ReLU(),
            # nn.Linear(2048, 1024),
            # nn.ReLU(),
            # nn.Linear(1024, 256),
            # nn.ReLU(),
            # nn.Linear(256, output)
        )
