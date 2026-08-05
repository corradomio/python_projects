import os
import argparse
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from edet.dataset import Resizer, Normalizer, Augmenter, collater
from edet.model import EfficientDet
# from tensorboardX import SummaryWriter
import shutil
import numpy as np
from tqdm.autonotebook import tqdm
from pathlib import Path



print(f"cuda: {torch.cuda.is_available()}")

# ---------------------------------------------------------------------------
# 1. Dataset
# ---------------------------------------------------------------------------

def clip(x, l: float):
    if x < 0: return 0.
    if x > 1: return l
    return x*l


def load_coords(coords_file: Path)-> list[list[float]]:
    coords_list = []
    with open(str(coords_file), mode="r") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.split(" ")
            coords = list(map(float, parts))
            coords_list.append(coords)
        pass
    pass
    # return np.array(coords_list)
    return coords_list
#end



class CocoDetectionDataset(Dataset):
    """
    Wraps a COCO-format dataset for use with torchvision's detection models.

    torchvision detection models expect each sample to be:
        image: FloatTensor[C, H, W], values in [0, 1]
        target: dict with
            boxes:    FloatTensor[N, 4]  (x1, y1, x2, y2) in absolute pixel coords
            labels:   Int64Tensor[N]     (1-indexed; 0 is reserved for background)
            image_id: Int64Tensor[1]
            area:     FloatTensor[N]
            iscrowd:  Int64Tensor[N]

    If your dataset isn't in COCO format, replace __getitem__ / __len__ with
    your own logic — the important part is returning the dict above.
    """
    def __init__(self, dataset_root: Path, type: str, transforms):
        images_dir = dataset_root / f"images/{type}"
        labels_dir = dataset_root / f"labels/{type}"

        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transforms = transforms

        self._image_files: list[Path] = []
        self._items: dict[int, tuple] = {}

        self.cat_id_to_label: dict[int, int] = {
            1: 1
        }
        self.label_to_name: dict[int, str] = {
            0: "background",
            1: "chicken"
        }

        for i, image_file in enumerate(images_dir.glob("*.jpg")):
            self._image_files.append(image_file)
        pass

    def __len__(self):
        return len(self._image_files)

    def __getitem__(self, idx: int):
        # tprint(f"... {idx}")
        if idx in self._items:
            return self._items[idx]

        image_file: Path = self._image_files[idx]
        image = Image.open(image_file).convert("RGB")
        W, H = image.size

        labels_file = self.labels_dir / (image_file.stem + ".txt")

        coords = load_coords(labels_file)

        boxes = []
        # labels = []
        # img_id = idx
        # areas = []
        # iscrowd = []

        for rec in coords:
            #   <class_id> <x_center> <y_center> <width> <height>
            #   RELATIVE: in range [0,1]
            class_id, xc, yc, w, h = rec

            if w == 0: w = 0.005
            if h == 0: h = 0.005

            dx = w/2
            dy = h/2

            x1 = clip(xc-dx, W)
            y1 = clip(yc-dy, H)
            x2 = clip(xc+dx, W)
            y2 = clip(yc+dy, H)

            boxes.append([x1, y1,x2, y2, 1])
            # labels.append(1)
            # areas.append(w*h*W*H)
            # iscrowd.append(0)
        pass

        # boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        # labels = torch.as_tensor(labels, dtype=torch.int64)
        # image_id = torch.tensor([img_id])
        # areas = torch.as_tensor(areas, dtype=torch.float32)
        # iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        # target = {
        #     "boxes": boxes,
        #     "labels": labels,
        #     "image_id": image_id,
        #     "area": areas,
        #     "iscrowd": iscrowd,
        # }
        target = {
            "img": np.array(image),
            "annot":np.array(boxes)
        }

        if self.transforms is not None:
            target = self.transforms(target)

        self._items[idx] = target
        return self._items[idx]
    # end

    def num_classes(self):
        return 2
# end


# def get_args():
#     parser = argparse.ArgumentParser(
#         "EfficientDet: Scalable and Efficient Object Detection implementation by Signatrix GmbH")
#     parser.add_argument("--image_size", type=int, default=512, help="The common width and height for all images")
#     parser.add_argument("--batch_size", type=int, default=8, help="The number of images per batch")
#     parser.add_argument("--lr", type=float, default=1e-4)
#     parser.add_argument('--alpha', type=float, default=0.25)
#     parser.add_argument('--gamma', type=float, default=1.5)
#     parser.add_argument("--num_epochs", type=int, default=500)
#     parser.add_argument("--test_interval", type=int, default=1, help="Number of epoches between testing phases")
#     parser.add_argument("--es_min_delta", type=float, default=0.0,
#                         help="Early stopping's parameter: minimum change loss to qualify as an improvement")
#     parser.add_argument("--es_patience", type=int, default=0,
#                         help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
#     parser.add_argument("--data_path", type=str, default="data/COCO", help="the root folder of dataset")
#     parser.add_argument("--log_path", type=str, default="tensorboard/efficientdet_chicken")
#     parser.add_argument("--saved_path", type=str, default="trained_models")
# 
#     args = parser.parse_args()
#     return args

class ThisResizer(Resizer):
    def __init__(self, common_size=512):
        self._common_size = common_size

    def __call__(self, sample):
        return super().__call__(sample, common_size=self._common_size)




def train(_):
    opt_batch_size = 1
    opt_data_path = Path(
        r"D:\Projects.ebtic\project.adafsa\pio_dataset"
        # r"D:\Users\corrado.mio\project.adafsa\pio_dataset"
    )
    opt_log_path = "logs"
    opt_saved_path = "trained_models"
    opt_lr = 1e-4
    opt_num_epochs = 500
    opt_test_interval = 1
    opt_es_min_delta = 0.0
    opt_es_patience = 0
    compound_coef = 4           # divide by 2^compound_coef
    use_cuda = torch.cuda.is_available()
    use_cuda = False

    num_gpus = 1
    if use_cuda:
        num_gpus = torch.cuda.device_count()
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)

    training_params = {
        "batch_size": opt_batch_size,
        "shuffle": True,
        "drop_last": True,
        "collate_fn": collater,
        # "num_workers": 12
    }

    test_params = {
        "batch_size": opt_batch_size,
        "shuffle": False,
        "drop_last": False,
        "collate_fn": collater,
        # "num_workers": 12
    }

    training_set = CocoDetectionDataset(
        dataset_root=opt_data_path,
        type="train",
        transforms=transforms.Compose([
            Normalizer(),
            # Augmenter(),
            ThisResizer(1536)
        ])
    )
    training_generator = DataLoader(training_set, **training_params)

    test_set = CocoDetectionDataset(
        dataset_root=opt_data_path,
        type="val",
        transforms=transforms.Compose([
            Normalizer(),
            ThisResizer(1536)
        ])
    )
    test_generator = DataLoader(test_set, **test_params)

    model = EfficientDet(
        num_classes=training_set.num_classes(),
        compound_coef=compound_coef,
        use_cuda=use_cuda
    )


    if os.path.isdir(opt_log_path):
        shutil.rmtree(opt_log_path)
    os.makedirs(opt_log_path)

    if not os.path.isdir(opt_saved_path):
        os.makedirs(opt_saved_path)

    # writer = SummaryWriter(opt_log_path)
    if use_cuda:
        model = model.cuda()
        model = nn.DataParallel(model)

    optimizer = torch.optim.Adam(model.parameters(), opt_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        # verbose=True
    )

    best_loss = 1e5
    best_epoch = 0
    model.train()

    # num_iter_per_epoch = len(training_generator)
    for epoch in range(opt_num_epochs):
        model.train()
        # if torch.cuda.is_available():
        #     model.module.freeze_bn()
        # else:
        #     model.freeze_bn()
        epoch_loss = []
        progress_bar = tqdm(training_generator)
        for iter, data in enumerate(progress_bar):
            try:
                optimizer.zero_grad()
                if use_cuda:
                    cls_loss, reg_loss = model([data['img'].cuda().float(), data['annot'].cuda()])
                else:
                    cls_loss, reg_loss = model([data['img'].float(), data['annot']])

                cls_loss = cls_loss.mean()
                reg_loss = reg_loss.mean()
                loss = cls_loss + reg_loss
                if loss == 0:
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
                optimizer.step()
                epoch_loss.append(float(loss.detach()))
                # total_loss = np.mean(epoch_loss)

                # progress_bar.set_description(
                #     'Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Batch loss: {:.5f} Total loss: {:.5f}'.format(
                #         epoch + 1, opt_num_epochs, iter + 1, num_iter_per_epoch, cls_loss, reg_loss, loss,
                #         total_loss))
                # writer.add_scalar('Train/Total_loss', total_loss, epoch * num_iter_per_epoch + iter)
                # writer.add_scalar('Train/Regression_loss', reg_loss, epoch * num_iter_per_epoch + iter)
                # writer.add_scalar('Train/Classfication_loss (focal loss)', cls_loss, epoch * num_iter_per_epoch + iter)

            except Exception as e:
                print(e)
                continue

        scheduler.step(np.mean(epoch_loss))

        if epoch % opt_test_interval == 0:
            model.eval()
            loss_regression_ls = []
            loss_classification_ls = []
            for iter, data in enumerate(test_generator):
                with torch.no_grad():
                    if use_cuda:
                        cls_loss, reg_loss = model([data['img'].cuda().float(), data['annot'].cuda()])
                    else:
                        cls_loss, reg_loss = model([data['img'].float(), data['annot']])

                    cls_loss = cls_loss.mean()
                    reg_loss = reg_loss.mean()

                    loss_classification_ls.append(float(cls_loss))
                    loss_regression_ls.append(float(reg_loss))

            cls_loss = np.mean(loss_classification_ls)
            reg_loss = np.mean(loss_regression_ls)
            loss = cls_loss + reg_loss

            print(
                'Epoch: {}/{}. Classification loss: {:1.5f}. Regression loss: {:1.5f}. Total loss: {:1.5f}'.format(
                    epoch + 1, opt_num_epochs, cls_loss, reg_loss,
                    np.mean(loss)))
            # writer.add_scalar('Test/Total_loss', loss, epoch)
            # writer.add_scalar('Test/Regression_loss', reg_loss, epoch)
            # writer.add_scalar('Test/Classfication_loss (focal loss)', cls_loss, epoch)

            if loss + opt_es_min_delta < best_loss:
                best_loss = loss
                best_epoch = epoch
                torch.save(model, os.path.join(opt_saved_path, "efficientdet_chicken.pth"))

                # dummy_input = torch.rand(opt_batch_size, 3, 512, 512)
                # if torch.cuda.is_available():
                #     dummy_input = dummy_input.cuda()
                # if isinstance(model, nn.DataParallel):
                #     model.module.backbone_net.model.set_swish(memory_efficient=False)
                #
                #     torch.onnx.export(model.module, dummy_input,
                #                       os.path.join(opt_saved_path, "efficientdet_chicken.onnx"),
                #                       verbose=False)
                #     model.module.backbone_net.model.set_swish(memory_efficient=True)
                # else:
                #     model.backbone_net.model.set_swish(memory_efficient=False)
                #
                #     torch.onnx.export(model, dummy_input,
                #                       os.path.join(opt_saved_path, "efficientdet_chicken.onnx"),
                #                       verbose=False)
                #     model.backbone_net.model.set_swish(memory_efficient=True)

            # Early stopping
            if epoch - best_epoch > opt_es_patience > 0:
                print("Stop training at epoch {}. The lowest loss achieved is {}".format(epoch, loss))
                break
    # writer.close()


if __name__ == "__main__":
    # opt = get_args()
    train({})
