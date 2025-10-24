from functools import lru_cache
from math import pi, pow

import torch
import torchvision
from PIL import Image
from path import Path as path
from torch.utils.data import Dataset

import stdlib.jsonx as jsonx


def sq(x: float) -> float: return x*x

def idof(s: str) -> str:
    p = s.index("-")
    return s[p+1:]


def surface(bboxes: list[list[float]]) -> float:
    s = 0.
    for bbox in bboxes:
        x1,y1,x2,y2 = bbox
        s += (x2-x1)*(y2-y1)
    return s


class WDBBoxDataset(Dataset):
    """
    WaterDrop for Bounding Box dataset
    """
    def __init__(self, image_dir: path, max_images: int = 5000):
        super().__init__()
        self.image_dir = image_dir
        self.max_images = max_images
        self.ToTensor = torchvision.transforms.ToTensor()

    def __len__(self):
        return self.max_images

    @lru_cache(maxsize=5000)
    def __getitem__(self, idx):
        im_tensor = None
        targets = {'bboxes': []}
        while len(targets['bboxes']) == 0:
            im_tensor, targets, _ = self.get_image(idx)
            idx += 1
        return (im_tensor, targets), targets
    # end

    def get_image(self, idx: int, dim4=False):
        sdir = idx // 1000
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
        if dim4:
            im_tensor = im_tensor[None, ...]
        return im_tensor, targets, image_path
# end


class WDContactAngleDataset(Dataset):
    """
    WaterDrop for Contact Angle dataset
    Return:
        image
        [xmin, ymin, xmax, ymax, contact_angle, drop_radius, drop_base, drop_height, drop_volume]

        xmin, xmax: [0,1] relative to width
        ymin, ymax: [0,1] relative to height
        contact_angle: [0,1] where 1 is 180         IN DEGREES
        drop_radius: [0,2]
        drop_base: [0,2]
        drop_height: [0,2]
        drop_volume: [0,1] relative to the sphere
    """
    def __init__(self, image_dir: path, max_images: int = 5000, crop: int = 0):
        super().__init__()
        self.image_dir = image_dir
        self.max_images = max_images
        self.crop = crop
        self.ToTensor = torchvision.transforms.ToTensor()
    # end

    # drop_base         [0, 2]  (diameter)
    # drop_height       [0, 2]  (diameter)

    def __len__(self):
        return self.max_images

    @lru_cache(maxsize=5000)
    def __getitem__(self, idx):
        return self.get_image(idx)
    # end

    def get_image(self, idx: int, dim4=False):
        json_path = ""
        try:
            sdir = idx // 1000
            json_path = self.image_dir / f"{sdir:04}/drop_scene-{idx:04}.json"
            image_path = self.image_dir / f"{sdir:04}/drop_scene-{idx:04}.png"
            im = Image.open(image_path)
            jdata = jsonx.load(json_path)

            width, height = jdata["image_size"]
            xmin, ymin, xmax, ymax = jdata["drop_bbox"]
            contact_angle = jdata["contact_angle"]
            drop_radius = jdata["drop_radius"]
            drop_base = jdata["drop_base"]
            drop_height = jdata["drop_height"]
            drop_volume_saved = jdata["drop_volume"]

            if self.crop > 0:
                # crop the image in square form
                if width > height:
                    shift = (width - height)//2
                    im = im.crop((shift, 0, shift+height, height))
                    xmin -= shift
                    xmax -= shift
                    width = height
                else:
                    shift = (height - width)//2
                    im = im.crop((0, shift, width, shift+width))
                    ymin -= shift
                    ymax -= shift
                    height = width

                im = im.resize((self.crop, self.crop))
                xmin = int(xmin * self.crop / width)
                ymin = int(ymin * self.crop / height)
                xmax = int(xmax * self.crop / width)
                ymax = int(ymax * self.crop / height)
                width = self.crop
                height = self.crop
            # end

            assert 0 <= contact_angle <= 180
            # THERE IS NOT A LIMIT on the max value for 'drop_radius', because
            # it depends on drop_base AND drop_height
            assert drop_radius > 0
            assert drop_base > 0
            assert drop_height > 0
            # assert 0 <= drop_base <= 4, f"drop_base={drop_base}"
            # assert 0 <= drop_height <= 2, f"drop_height={drop_height}"
            # BECAUSE there is not a limit on the radius, there is not a LIMIT on the
            # volume. Than, HOW to normalize the volume?
            # assert 0 <= drop_volume_saved <= self.VOLUME_SPHERE_RADIUS_1

            drop_volume = pi*pow(drop_height, 2)*(drop_radius - drop_height/3)
            sphere_volume = 4/3*pi*pow(drop_radius, 3)

            assert drop_volume_saved <= sphere_volume and drop_volume > 0

            im_tensor = self.ToTensor(im)
            info = torch.tensor([
                # xmin/width, ymin/height,
                # xmax/width, ymax/height,
                contact_angle/180,
                # drop_radius,
                drop_base/2,
                drop_height/2,
                drop_volume/sphere_volume
            ])
            if dim4:
                im_tensor = im_tensor[None, ...]
            return im_tensor, info
        except AssertionError as e:
            print(f"json={json_path}", e)
            raise e
# end

