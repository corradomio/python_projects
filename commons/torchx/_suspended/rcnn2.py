import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv


#
# Conv2d
#   [N,C_in,H,W]
#   [N,C_out,H_out,W_out]


class VGG(nn.Module):
    ''' VGG-16 Network with only convolutions '''

    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.feature_extractor(x)


class RPN(nn.Module):
    ''' Region Proposal Network '''

    def __init__(self, k=9):
        super().__init__()
        self.preprocess = nn.Sequential(
            nn.Conv2d(512, 512, 3),
            nn.ReLU()
        )
        self.classifier = nn.Conv2d(512, 18, 1)
        self.regressor = nn.Conv2d(512, 36, 1)

    def forward(self, x):
        x = self.preprocess(x)
        obj_preds = self.classifier(x)
        bbox_preds = self.regressor(x)
        return obj_preds, bbox_preds


class FastRCNN(nn.Module):
    ''' Fast R-CNN Network '''

    def __init__(self):
        super().__init__()
        # Note: 7 is chosen arbitrarily for now
        self.feature_extractor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )
        # 4 is refined bounding boxes for detecting starfish
        self.regressor = nn.Linear(1024, 4)
        # 2 is for whether the box contains the starfish or not
        # might not need this since RPN detects object or not
        self.classifier = nn.Linear(1024, 2)

    def forward(self, x, proposals):
        # tv.ops.roi_pool(Tensor[N, C, H, W], Tensor[K, 5] | List[Tensor[L, 4]])
        x = tv.ops.roi_pool(x, proposals, output_size=7)
        x = self.feature_extractor(x)
        bbox_preds = self.regressor(x)
        class_preds = self.classifier(x)
        return class_preds, bbox_preds


class FasterRCNN(nn.Module):
    def __init__(self, channels_last=True):
        super().__init__()
        self.VGG = VGG()
        self.RPN = RPN()
        self.FastRCNN = FastRCNN()
        self.channels_last = channels_last

    def forward(self, x):
        # (N, W, H, C) -> (N, C, W, H)
        if self.channels_last:
            x = torch.permute(x, [0, 3, 1, 2])

        x = self.VGG(x)
        # proposal = obj_preds, bbox_preds
        #            [1,18,1,1] [1,36,1,1]
        proposals = self.RPN(x)
        outputs = self.FastRCNN(x, proposals)
        return proposals, outputs


class MultiTaskLoss(nn.Module):
    ''' Computes combined loss from '''

    def __init__(self, n_cls, n_reg, l):
        super().__init__()
        self.n_cls = n_cls
        self.n_reg = n_reg
        self.l = l

    def forward(self, predictions, targets):
        bbox_preds = predictions[0]
        class_preds = predictions[1]
        bbox_truth = targets[0]
        class_truth = targets[1]
        return (1 / self.n_cls) * F.binary_cross_entropy_with_logits(class_preds, class_truth) \
            + (self.l / self.n_reg) * (class_truth * F.smooth_l1_loss(bbox_preds, bbox_truth))
# end
