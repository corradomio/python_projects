import pickle

import torch
from torch.optim.lr_scheduler import MultiStepLR
from path import Path as path

from faster_rcnn_common import WDContactAngleDataset
from skorchx import NeuralNetRegressor
from skorchx.callbacks import OnEvent
from torchx.nn.cvision import AlexNetV2

IMAGES_DIR = path(r"E:/Datasets/WaterDrop/orig")


waterdrop_dataset = WDContactAngleDataset(image_dir=IMAGES_DIR, max_images=5000, crop=224)


nnet = NeuralNetRegressor(
    module=AlexNetV2,
    module__output=4,
    #
    optimizer=torch.optim.SGD,
    optimizer__weight_decay=0.0005, #5E-4,
    optimizer__momentum=0.9,
    #
    max_epochs=200,
    batch_size=128,
    lr=0.001,
    #
    scheduler=MultiStepLR,
    scheduler__milestones=[60,120],
    scheduler__gamma=0.1,
    #
    callbacks=[
        OnEvent()
    ],
    device = 'cuda',
)


def main():
    data = waterdrop_dataset[0]

    nnet.fit(waterdrop_dataset)

    with open('waterdrop-alexnet.pkl', 'wb') as f:
        pickle.dump(nnet, f)

    pass



if __name__ == "__main__":
    main()
