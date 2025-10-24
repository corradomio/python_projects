import matplotlib.pyplot as plt
import torch.nn as nn
from path import Path as path

from skorchx import NeuralNetClassifier
from torchx.nn import SegNet
from wddataset import WaterDropDataset

IMAGES_DIR = path(r"E:\Datasets\WaterDrop\orig")


nnet = NeuralNetClassifier(
    module=SegNet,
    module__in_channels=3,
    module__num_classes=4,
    #
    max_epochs=60,
    lr=0.05,
    batch_size=16,  # 128
    #
    criterion=nn.CrossEntropyLoss(),
    #
    device="cuda"
)
# in: (B, 460, 640, 3)  float32
# out (B, 480, 640)     int64
def main():
    wdds = WaterDropDataset(IMAGES_DIR, start=0, count=1000)

    X, y = wdds.get_image(1000)
    nnet.initialize()
    t = nnet.predict(X)

    nnet.fit(wdds)

    X, y = wdds.get_image(1000)
    t = nnet.predict(X)

    plt.imshow(y[0])
    plt.show()
    plt.imshow(t[0])
    plt.show()
# end



if __name__ == "__main__":
    main()


