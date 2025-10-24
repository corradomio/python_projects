import pickle

import torch
from torch.optim.lr_scheduler import MultiStepLR
from path import Path as path
from torchvision.models import VGG16_Weights

from faster_rcnn_common import WDContactAngleDataset
from skorchx import NeuralNetRegressor
from skorchx.callbacks import OnEvent
from torchx.nn.cvision.vgg import VGG16
# from torchx.nn.cvision.vggnet import VGG16


IMAGES_DIR = path(r"E:/Datasets/WaterDrop/orig2")


waterdrop_dataset = WDContactAngleDataset(image_dir=IMAGES_DIR, max_images=5000, crop=224)


nnet = NeuralNetRegressor(
    module=VGG16,
    module__weights=None,
    module__bias=False,
    module__output=4,
    #
    optimizer=torch.optim.SGD,
    optimizer__weight_decay=0.0005, #5E-4,
    optimizer__momentum=0.9,
    # optimizer=torch.optim.NAdam,
    #
    max_epochs=200,
    batch_size=50,
    lr=0.001,
    #
    scheduler=MultiStepLR,
    scheduler__milestones=[50, 100, 150],
    # scheduler__milestones=[75, 150],
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

    with open('waterdrop2-vgg.pkl', 'wb') as f:
        pickle.dump(nnet, f)

    pass


if __name__ == "__main__":
    main()
