# -*- coding: utf-8 -*-
"""
.. codeauthor:: Matteo Sodano <matteo.sodano@igg.uni-bonn.de>
"""

import os

import imageio
import cv2 as cv
import numpy as np
import scipy.ndimage as spni
import scipy.ndimage.measurements as m
import torch

from .scannet import SCANNETBase
from ..dataset_base import DatasetBase


class ScanNet(SCANNETBase, DatasetBase):
    def __init__(self,
                 data_dir=None,
                 split='train',
                 with_input_orig=False,
                 debug=False):
        super(ScanNet, self).__init__()

        self.debug = debug
        self._n_classes = self.N_CLASSES
        self._cameras = ['camera1']
        assert split in self.SPLITS, \
            f'parameter split must be one of {self.SPLITS}, got {split}'
        self._split = split
        self._with_input_orig = with_input_orig

        if data_dir is not None:
            data_dir = os.path.expanduser(data_dir)
            self._data_dir = data_dir
            self.scans_list = self.load_file_lists(self.split)
        else:
            print(f"Loaded {self.__class__.__name__} dataset without files")
        self.images, self.img_dir = [], []
        self.depths, self.depth_dir = [], []
        self.labels, self.label_dir = [], []
        self.instances, self.instance_dir = [], []
        if data_dir is not None:
            for scan in self.scans_list:
                color_fold = scan + '/color/'
                depth_fold = scan + '/depth/'
                label_fold = scan + '/label-filt/'
                instance_fold = scan + '/instance-filt/'
                color_images = os.listdir(color_fold)
                depth_images = os.listdir(depth_fold)
                self.img_dir.append(color_fold)
                self.depth_dir.append(depth_fold)
                self.label_dir.append(label_fold)
                self.instance_dir.append(instance_fold)
                self.images += [os.path.join(color_fold, image) for image in color_images]
                self.depths += [os.path.join(depth_fold, image) for image in depth_images]

            self.images.sort()
            self.depths.sort()

            for file in self.depths:
                semantic_label = file.replace('depth', 'label-filt')
                instance_label = file.replace('depth', 'instance-filt')
                final_part = semantic_label.split('/')[-1]
                name = str(int(final_part.split('.')[0]))
                semantic_label = semantic_label.replace(final_part, name+'.png')
                instance_label = instance_label.replace(final_part, name+'.png')
                self.labels.append(semantic_label)
                self.instances.append(instance_label)
        else:
            print(f"Loaded {self.__class__.__name__} dataset without files")

        self.list_of_semantic_labels = self.labels_numbers
        self._class_names = self.labels_names
        self._class_colors = np.array(self.CLASS_COLORS, dtype='uint8')

        self._depth_mean = 0
        self._depth_std = 1

    @property
    def cameras(self):
        return self._cameras

    @property
    def class_names(self):
        return self._class_names

    @property
    def class_names_without_void(self):
        return self._class_names[1:]

    @property
    def class_colors(self):
        return self._class_colors

    @property
    def class_colors_without_void(self):
        return self._class_colors[1:]

    @property
    def n_classes(self):
        return self._n_classes

    @property
    def n_classes_without_void(self):
        return self._n_classes

    @property
    def split(self):
        return self._split

    @property
    def depth_mean(self):
        return self._depth_mean

    @property
    def depth_std(self):
        return self._depth_std

    @property
    def source_path(self):
        return os.path.abspath(os.path.dirname(__file__))

    @property
    def with_input_orig(self):
        return self._with_input_orig

    def load_name(self, idx):
        return self.images[idx]

    def compute_center_coordinates(self, center_tensor):
        center_coordinates = []
        centers = torch.where(center_tensor >= 1. - 1e-3)  # extract coordinates of points == 1
        xc, yc = centers[0], centers[1]

        assert len(xc) == len(yc)

        for i in range(len(xc)):
            # add tuple to list
            cen = (int(xc[i]), int(yc[i]))
            center_coordinates.append(cen)

        return center_coordinates

    def load_center(self, idx, sample):
        semantic = sample['label'].numpy()
        instance = sample['instance'].numpy() * (semantic != 0).astype(int)
        labels = np.unique(instance)
        labels = labels[labels > 0].astype(int)
        center_mask = np.zeros_like(instance, dtype=float)
        for label in labels:
            img_label = (instance == label).astype(int)
            com = m.center_of_mass(img_label)
            sem_label = semantic[int(com[0]), int(com[1])]
            if sem_label == 1. or sem_label == 2. or sem_label > len(self.list_of_semantic_labels):
                continue
            tmp = np.zeros_like(center_mask, dtype=float)
            tmp[int(com[0]), int(com[1])] = 1.
            tmp = spni.gaussian_filter(tmp, 5)
            normalized_mask = cv.normalize(tmp, None, 0, 1, cv.NORM_MINMAX)
            center_mask = np.maximum(center_mask, normalized_mask)
        center_mask = torch.from_numpy(center_mask).float()
        return center_mask

    def load_image(self, idx):
        color_img = self.images[idx]
        color_mean = np.array([0., 0., 0.])
        color_std = np.array([1., 1., 1.])
        image = np.array(imageio.imread(color_img))
        image = (image - color_mean)/color_std
        return image

    def load_depth(self, idx):
        depth_img = self.depths[idx]
        depth = np.array(imageio.imread(depth_img)).astype(np.float32) / 1000.0
        return depth

    def load_instance(self, idx, sample):
        semantic = np.copy(sample['label'])
        semantic[semantic == 1.] = 0.   # wall
        semantic[semantic == 2.] = 0.   # floor
        instance_img = self.instances[idx]
        instance_label = np.array(imageio.imread(instance_img)).astype(np.uint8)
        instance_label = instance_label * (semantic > 0).astype(int)
        return instance_label, semantic

    def load_label(self, idx):
        semantic_img = self.labels[idx]
        semantic_label = np.array(imageio.imread(semantic_img)).astype(np.uint8)
        for index in np.unique(semantic_label):
            if index not in self.labels_mapping:
                semantic_label[semantic_label == index] = 0
            else:
                mapped_label = self.labels_mapping[index]
                semantic_label[semantic_label == index] = self.list_of_semantic_labels.index(mapped_label)
        return semantic_label

    def number_of_samples(self):
        samples = self.__len__()
        return samples

    def load_file_lists(self, split):
        def _get_filepath(filename):
            return os.path.join(self._data_dir, filename)

        scans = []
        if split == 'train':
            train_scans = _get_filepath('train.txt')
            scans = self.list_and_dict_from_file(train_scans)
        elif split == 'test':
            test_scans = _get_filepath('test.txt')
            scans = self.list_and_dict_from_file(test_scans)

        return scans

    def list_and_dict_from_file(self, filepath):
        with open(filepath, 'r') as f:
            file_list = f.read().splitlines()
        return file_list

    def __len__(self):
        if self.debug:
            return self.debug
        return len(self.images)
