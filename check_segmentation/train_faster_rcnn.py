import pickle

import torch
from path import Path as path
from torch.optim.lr_scheduler import MultiStepLR

from faster_rcnn_common import WDBBoxDataset
from skorchx import NeuralNet
from skorchx.callbacks import OnEvent
from torchx.nn.cvision import FasterRCNN, FasterRCNNLoss
from torchx.nn.cvision.bbox import save_image

IMAGES_DIR = path(r"E:/Datasets/WaterDrop/orig")


waterdrop_dataset = WDBBoxDataset(image_dir=IMAGES_DIR, max_images=5000)


nnet = NeuralNet(
    module=FasterRCNN,
    # module__channels_last=True,
    module__in_channels=3,
    module__num_classes=2,
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
    lr=0.001,
    #
    scheduler=MultiStepLR,
    scheduler__milestones=[8, 16],
    scheduler__gamma=0.1,
    #
    callbacks=[
        OnEvent()
    ],
    device = 'cuda',
)


def main():
    # data = waterdrop_dataset[0]

    nnet.fit(waterdrop_dataset)

    with open('waterdrop-fastercnn.pkl', 'wb') as f:
        pickle.dump(nnet, f)

    nnet.module_.roi_head.low_score_threshold = 0.7

    im_tensor, targets, image_path = waterdrop_dataset.get_image(7002, True)
    rpn_output, frcnn_output = nnet.predict(im_tensor)
    bboxes = frcnn_output['bboxes']
    labels = frcnn_output['labels']
    scores = frcnn_output['scores']

    save_image(image_path, "test.png", frcnn_output, {0: "background", 1: "waterdrop"})

    pass



if __name__ == "__main__":
    main()
