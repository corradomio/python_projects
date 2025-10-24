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


def main():
    data = waterdrop_dataset[0]

    with open('waterdrop-enet.pkl', 'rb') as f:
        nnet: NeuralNetRegressor = pickle.load(f)

    for i in range(10):
        im_tensor, targets = waterdrop_dataset.get_image(7000+i, True)
        predictions = nnet.predict(im_tensor)
        print(targets)
        print(predictions)
        print("---")
pass


if __name__ == "__main__":
    main()
