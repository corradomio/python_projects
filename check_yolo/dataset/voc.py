import glob
import os
import random
from functools import lru_cache

import torch
import torchvision
from pprint import pprint
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
import xml.etree.ElementTree as ET


def load_images_and_anns(im_dir, ann_dir, label2idx):
    r"""
    Method to get the xml files and for each file
    get all the objects and their ground truth detection
    information for the dataset
    :param im_dir: Path of the images
    :param ann_dir: Path of annotation xmlfiles
    :param label2idx: Class Name to index mapping for dataset
    :return:
    """
    im_infos = []
    for ann_file in tqdm(glob.glob(os.path.join(ann_dir, '*.xml'))):
        im_info = {}
        im_info['img_id'] = os.path.basename(ann_file).split('.xml')[0]
        im_info['filename'] = os.path.join(im_dir, '{}.jpg'.format(im_info['img_id']))
        ann_info = ET.parse(ann_file)
        root = ann_info.getroot()
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        im_info['width'] = width
        im_info['height'] = height
        detections = []
        
        for obj in ann_info.findall('object'):
            det = {}
            label = label2idx[obj.find('name').text]
            bbox_info = obj.find('bndbox')
            bbox = [
                int(float(bbox_info.find('xmin').text))-1,
                int(float(bbox_info.find('ymin').text))-1,
                int(float(bbox_info.find('xmax').text))-1,
                int(float(bbox_info.find('ymax').text))-1
            ]
            det['label'] = label
            det['bbox'] = bbox
            detections.append(det)
        im_info['detections'] = detections
        im_infos.append(im_info)
    print('Total {} images found'.format(len(im_infos)))
    return im_infos


def pil_resize(im: Image.Image, bboxes: list[list[int]], labels: list[int], new_size: tuple[int, int], n_targets: int) \
        -> tuple[Image.Image, list[list[int]], list[int]]:
    width, height = im.size
    new_width, new_height = new_size

    def bbox_resize(bbox):
        x1, y1, x2, y2 = bbox
        return [
            x1 * new_width // width,
            y1 * new_height // height,
            x2 * new_width // width,
            y2 * new_height // height
        ]
    # end

    new_im = im.resize(new_size)
    new_bboxes = [
        bbox_resize(bbox)
        for bbox in bboxes
    ]
    new_labels = labels[:]
    while len(new_bboxes) < n_targets:
        new_bboxes.append([0,0,1,1])
        new_labels.append(-1)
    return new_im, new_bboxes, new_labels


class VOCDataset(Dataset):
    def __init__(self, split, im_dir, ann_dir):
        self.split = split
        self.im_dir = im_dir
        self.ann_dir = ann_dir
        classes = [
            'person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
            'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
            'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor'
        ]
        classes = sorted(classes)
        classes = ['background'] + classes
        self.label2idx = {classes[idx]: idx for idx in range(len(classes))}
        self.idx2label = {idx: classes[idx] for idx in range(len(classes))}
        pprint(self.idx2label)
        self.images_info = load_images_and_anns(im_dir, ann_dir, self.label2idx)
        self.ToTensor = torchvision.transforms.ToTensor()
    
    def __len__(self):
        return len(self.images_info)

    @lru_cache(maxsize=1024)
    def __getitem__(self, index):
        im_info = self.images_info[index]
        im = Image.open(im_info['filename'])
        bboxes = [detection['bbox'] for detection in im_info['detections']]
        labels = [detection['label'] for detection in im_info['detections']]

        # im, bboxes, labels = pil_resize(im, bboxes, labels, (500, 500), 42)

        to_flip = False
        if self.split == 'train' and random.random() < 0.5:
            to_flip = True
            im = im.transpose(Image.FLIP_LEFT_RIGHT)

        im_tensor = self.ToTensor(im)

        targets = {}
        targets['bboxes'] = torch.as_tensor(bboxes)
        targets['labels'] = torch.as_tensor(labels)

        if to_flip:
            for idx, box in enumerate(targets['bboxes']):
                x1, y1, x2, y2 = box
                w = x2-x1
                im_w = im_tensor.shape[-1]
                x1 = im_w - x1 - w
                x2 = x1 + w
                targets['bboxes'][idx] = torch.as_tensor([x1, y1, x2, y2])
        return im_tensor, targets, im_info['filename']



def main():

    voc = VOCDataset('train',
                     im_dir='E:/Datasets/VOC2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/JPEGImages',
                     ann_dir='E:/Datasets/VOC2007/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/Annotations')

    n = len(voc)
    n_bboxes = 0
    for i in range(n):
        im, targets, filename = voc[i]
        t_bboxes = len(targets['bboxes'])
        if t_bboxes > n_bboxes:
            n_bboxes = t_bboxes
            print(n_bboxes, filename)
    print(n_bboxes)



if __name__ == '__main__':
    main()
