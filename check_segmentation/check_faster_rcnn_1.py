from functools import lru_cache

import torch
import torch
import torchvision
from PIL import Image
from path import Path as path
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import Dataset, DataLoader

import skorch.callbacks
import stdlib.jsonx as jsonx
from skorchx import NeuralNet
from torchx.nn.cvision import FasterRCNN, FasterRCNNLoss

IMAGES_DIR = path(r"E:/Datasets/WaterDrop/cropped")
# IMAGES_DIR = r"images"


def idof(s: str) -> str:
    p = s.index("-")
    return s[p+1:]


def surface(bboxes: list[list[float]]) -> float:
    s = 0.
    for bbox in bboxes:
        x1,y1,x2,y2 = bbox
        s += (x2-x1)*(y2-y1)
    return s


class WaterDropDataset(Dataset):
    def __init__(self, image_dir: path, max_images: int = 1000):
        super().__init__()
        self.image_dir = image_dir
        self.max_images = max_images
        self.ToTensor = torchvision.transforms.ToTensor()

    def __len__(self):
        return self.max_images

    @lru_cache(maxsize=1000)
    def __getitem__(self, idx):
        sdir = idx//1000
        json_path = self.image_dir / f"{sdir:04}/drop_scene-{idx:04}.json"
        image_path = self.image_dir / f"{sdir:04}/drop_scene-{idx:04}.png"
        jdata = jsonx.load(json_path)
        im = Image.open(image_path)
        bboxes = [jdata["drop_bbox"]]
        labels = [1]

        im_tensor = self.ToTensor(im)
        targets = {
            "bboxes": torch.as_tensor(bboxes),
            "labels": torch.as_tensor(labels)
        }

        # return im_tensor, targets, str(image_path)
        return (im_tensor, targets), targets
    # end

    def get_image(self, idx):
        (im_tensor, _), _ = self.__getitem__(idx)
        im_tensor = im_tensor[None, ...]
        return im_tensor

# end

waterdrop_dataset = WaterDropDataset(image_dir=IMAGES_DIR, max_images=1000)

# train_dataset = DataLoader(waterdrop_dataset, batch_size=1, shuffle=False, num_workers=4)

# faster_rcnn_model = FasterRCNN(channels_last=True, num_classes=2)

# optimizer = torch.optim.SGD(
#     lr=0.001,
#     params=filter(lambda p: p.requires_grad, faster_rcnn_model.parameters()),
#     weight_decay=5E-4,
#     momentum=0.9
# )

# scheduler = MultiStepLR(optimizer, milestones=[12, 16], gamma=0.1)


# loss_fn = FasterRCNNLoss()

# oparams = list(filter(lambda p: p.requires_grad, faster_rcnn_model.parameters()))


nnet = NeuralNet(
    module=FasterRCNN,
    # module__channels_last=True,
    module__input_shape=(3, 96, 96),
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
    max_epochs=1,
    batch_size=1,
    lr=0.1,
    #
    scheduler=MultiStepLR,
    scheduler__milestones=[4, 6, 8],
    scheduler__gamma=0.1,
    #
    # callbacks=[
    #     skorch.callbacks.LRScheduler(
    #         policy=MultiStepLR,
    #         milestones=[12, 16],
    #         gamma=0.1,
    #     )
    # ],
    device = 'cuda',
)

def main():
    # data = waterdrop_dataset[0]

    im_tensor = waterdrop_dataset.get_image(1000)

    # nnet.fit(waterdrop_dataset)
    nnet.initialize()
    #
    predicted = nnet.predict(im_tensor)

    pass



if __name__ == "__main__":
    main()
