import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
from torch import Tensor
from path import Path as path
from torchinfo import summary
from torch.optim.lr_scheduler import StepLR
from skorchx import NeuralNetClassifier
from torchx.nn import UNet, SegNet
from skorch.callbacks import LRScheduler
from wddataset import WaterDropDataset
import matplotlib.pyplot as plt


IMAGES_DIR = path(r"E:\Datasets\WaterDrop\cropped")


def idof(s: str) -> str:
    p = s.index("-")
    return s[p+1:]

#
# torch.optim
#   Adadelta      Implements Adadelta algorithm.
#   Adafactor     Implements Adafactor algorithm.
#   Adagrad       Implements Adagrad algorithm.
#   Adam          Implements Adam algorithm.
#   AdamW         Implements AdamW algorithm, where weight decay does not accumulate in the momentum nor variance.
#   SparseAdam    SparseAdam implements a masked version of the Adam algorithm suitable for sparse gradients.
#   Adamax        Implements Adamax algorithm (a variant of Adam based on infinity norm).
#   ASGD          Implements Averaged Stochastic Gradient Descent.
#   LBFGS         Implements L-BFGS algorithm.
#   NAdam         Implements NAdam algorithm.
#   RAdam         Implements RAdam algorithm.
#   RMSprop       Implements RMSprop algorithm.
#   Rprop         Implements the resilient backpropagation algorithm.
#   SGD           Implements stochastic gradient descent (optionally with momentum).


def dice_loss(pred, target, smooth=1., dim=2):
    # [B,C,W,H] -> dim = 2
    # [B,W,H,C] -> dim = 1
    pred = F.sigmoid(pred)
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=dim).sum(dim=dim)
    union = pred.sum(dim=dim).sum(dim=dim) + target.sum(dim=dim).sum(dim=dim)
    # [B,C]
    loss = (1 - ((2. * intersection + smooth) / (union + smooth)))
    # real
    return loss.mean()


class SegmentationLoss(nn.Module):
    def __init__(self, bce_weight=0.5, smooth=1., classes_last=False):
        super().__init__()
        self.bce_weight = bce_weight
        self.smooth = smooth
        self.classes_last = classes_last

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        bce_weight = self.bce_weight
        smooth = self.smooth
        bce = F.binary_cross_entropy_with_logits(pred, target)
        dice = dice_loss(pred, target, smooth, 1 if self.classes_last else 2)
        loss = bce * bce_weight + dice * (1 - bce_weight)
        return loss
# end


unet = UNet(in_channels=3, num_classes=4, channels_last=True, classes_last=True)
# segnet = SegNet(in_channels=3, num_classes=4, channels_last=False)
# summary(unet, input_size=(1,3,640,480))

BATCH_SIZE=64

net = NeuralNetClassifier(
    module=unet,
    # module=segnet,
    max_epochs=64,
    criterion=SegmentationLoss,
    criterion__bce_weight=0.49,
    criterion__smooth=0.9,
    criterion__classes_last=True,
    # optimizer=opt.SGD,
    # lr=0.05,
    optimizer=opt.Adam,
    lr=0.001,
    batch_size=BATCH_SIZE,  # 128
    # train_split=None,
    # iterator_train__shuffle=True,
    device="cuda",
    callbacks=[
        LRScheduler(
            policy="StepLR",
            step_every='epoch',
            step_size=BATCH_SIZE//8,
            gamma=0.5,
        )
    ]
)


def main():
    wdds = WaterDropDataset(IMAGES_DIR, start=0, count=1000)

    net.fit(wdds)

    for i in range(1, 10):

        X, y = wdds.get_image(i*1000)
        t = net.predict(X, classes_last=True)
        t: np.ndarray = t.numpy()
        # t = t.swapaxes(-1, -2)

        plt.imshow(y[0])
        plt.show()
        plt.imshow(t[0])
        plt.show()

# end



if __name__ == "__main__":
    main()


