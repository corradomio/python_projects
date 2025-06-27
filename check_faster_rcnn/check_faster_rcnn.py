import torch
from skorchx import NeuralNet
from skorchx.callbacks import OnEvent
from torch.optim.lr_scheduler import MultiStepLR
from torchx.nn.cvision import FasterRCNN, FasterRCNNLoss
from voc import VOCDataset
import pickle


def main():

    voc_ds = VOCDataset(
        'train',
        im_dir='E:/Datasets/VOC2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages',
        ann_dir='E:/Datasets/VOC2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations'
    )

    net = NeuralNet(
        module=FasterRCNN,
        module__in_channels=3,
        module__num_classes=21,
        # module__im_channels=3,
        module__aspect_ratios=[0.5, 1, 2],
        module__scales=[128, 256, 512],
        module__min_im_size=600,
        module__max_im_size=1000,
        module__backbone_out_channels=512,
        module__fc_inner_dim=1024,
        module__rpn_bg_threshold=0.3,
        module__rpn_fg_threshold=0.7,
        module__rpn_nms_threshold=0.7,
        module__rpn_train_prenms_topk=12000,
        module__rpn_test_prenms_topk=6000,
        module__rpn_train_topk=2000,
        module__rpn_test_topk=300,
        module__rpn_batch_size=256,
        module__rpn_pos_fraction=0.5,
        module__roi_iou_threshold=0.5,
        module__roi_low_bg_iou=0.0,  # increase it to 0.1 for hard negative
        module__roi_pool_size=7,
        module__roi_nms_threshold=0.3,
        module__roi_topk_detections=100,
        module__roi_score_threshold=0.05,
        module__roi_batch_size=128,
        module__roi_pos_fraction=0.25,
        #
        criterion=FasterRCNNLoss,
        #
        optimizer=torch.optim.SGD,
        # optimizer__params=filter(lambda p: p.requires_grad, faster_rcnn_model.parameters()),
        optimizer__weight_decay=5E-4,
        optimizer__momentum=0.9,
        # optimizer_params=dict(
        #     # lr=0.001,
        #     params=filter(lambda p: p.requires_grad, faster_rcnn_model.parameters()),
        #     weight_decay=5E-4,
        #     momentum=0.9
        # ),
        #
        max_epochs=20,
        batch_size=1,
        lr=0.1,
        #
        scheduler=MultiStepLR,
        scheduler__milestones=[12, 16],
        scheduler__gamma=0.1,
        #
        callbacks=[
            OnEvent()
        ],
        device='cuda',
    )

    net.fit(voc_ds)

    with open('faster-rcnn.pkl', 'wb') as f:
        pickle.dump(net, f)

    pass


if __name__ == "__main__":
    main()
