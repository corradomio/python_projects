# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re

import os.path as osp
import os

from .bases import BaseImageDataset
from collections import defaultdict
import pickle


class ReID780Dataset(object):
    """
    ReID780 dataset provided for the final term project of the course COL780, IITD
    
    Dataset statistics:
    # identities: 114
    # images: 992 (train) + 192 (query) + 192 (gallery)
    """

    def __init__(self, root='data', **kwargs):
        self.dataset_dir = root
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'val', 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'val', 'gallery')

        self._check_before_run()

        train, num_train_pids, num_train_imgs, num_train_cams = self._process_dir(self.train_dir, relabel=True)
        query, num_query_pids, num_query_imgs, num_query_cams = self._process_dir(self.query_dir, relabel=False)
        gallery, num_gallery_pids, num_gallery_imgs, num_gallery_cams = self._process_dir(self.gallery_dir, relabel=False)
        num_total_pids = num_train_pids + num_query_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_gallery_imgs

        print("=> ReID780 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("  gallery  | {:5d} | {:8d}".format(num_gallery_pids, num_gallery_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_cams = num_train_cams
        self.num_train_pids = num_train_pids
        self.num_train_imgs = num_train_imgs
        self.num_train_vids = 1

        self.num_query_cams = num_query_cams
        self.num_query_pids = num_query_pids
        self.num_query_imgs = num_query_imgs
        self.num_query_vids = 1
        
        self.num_gallery_cams = num_gallery_cams
        self.num_gallery_pids = num_gallery_pids
        self.num_gallery_imgs = num_gallery_imgs
        self.num_gallery_vids = 1

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        dataset = []
        p_ids = []
        cam_ids = []
        for p_id in os.listdir(dir_path):
            if ".DS_Store" in p_id:
                continue
            p_ids.append(p_id)
            image_names = os.listdir(osp.join(dir_path, p_id))
            image_names.sort()
            for cam_id, image_name in enumerate(image_names):
                img_path = osp.join(dir_path, p_id, image_name)
                dataset.append((img_path, p_id, cam_id, 1))
        pid2label = {p_id: i for i, p_id in enumerate(p_ids)}
        if relabel:
            new_dataset = []
            for data in dataset:
                img_path, p_id, cam_id, _ = data
                label = pid2label[p_id]
                new_dataset.append((img_path, label, cam_id, 1))
                cam_ids.append(cam_id)
            dataset = new_dataset

        num_pids = len(set(p_ids))
        num_imgs = len(dataset)
        num_cams = len(set(cam_ids))
        return dataset, num_pids, num_imgs, num_cams
