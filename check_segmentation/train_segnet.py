import torch.nn as nn
from path import Path as path
from torchinfo import summary
from skorchx import NeuralNetClassifier
from torchx.nn import SegNet
from wddataset import WaterDropDataset
import matplotlib.pyplot as plt


IMAGES_DIR = path(r"E:\Datasets\WaterDrop\cropped")


def idof(s: str) -> str:
    p = s.index("-")
    return s[p+1:]


segnet = SegNet(in_channels=1, num_classes=4)
summary(segnet, input_size=(1,640,480, 1))


net = NeuralNetClassifier(
    segnet,
    max_epochs=60,
    criterion=nn.CrossEntropyLoss(),
    lr=0.05,
    batch_size=16,  # 128
    train_split=None,
    iterator_train__shuffle=True,
    device="cuda"
)


def main():
    wdds = WaterDropDataset(IMAGES_DIR, start=0, count=1000)

    net.fit(wdds)

    X, y = wdds.get_image(1000)
    t = net.predict(X)

    plt.imshow(y[0])
    plt.show()
    plt.imshow(t[0])
    plt.show()
# end



if __name__ == "__main__":
    main()


