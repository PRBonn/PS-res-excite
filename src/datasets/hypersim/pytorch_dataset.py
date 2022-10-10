# -*- coding: utf-8 -*-
import os
import h5py
import imageio
import cv2 as cv
import numpy as np
import scipy.ndimage as spni
import scipy.ndimage.measurements as m
import torch

from .hypersim import HYPERSIMBase
from ..dataset_base import DatasetBase


class HyperSim(HYPERSIMBase, DatasetBase):
    def __init__(self,
                 data_dir=None,
                 split='train',
                 with_input_orig=False,
                 debug=False):
        super(HyperSim, self).__init__()

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

        self.images = []
        self.depths = []
        self.labels = []
        self.instances = []
        if data_dir is not None:
            for scan in self.scans_list:
                subfolders = os.listdir(os.path.join(scan, 'images'))
                for subfolder in subfolders:
                    files = os.listdir(os.path.join(scan, 'images', subfolder))
                    for file in files:
                        if 'color' in file:
                            self.images.append(os.path.join(scan, 'images', subfolder, file))
                        elif 'depth' in file:
                            self.depths.append(os.path.join(scan, 'images', subfolder, file))
                        elif 'instance' in file:
                            self.instances.append(os.path.join(scan, 'images', subfolder, file))
                        else:
                            self.labels.append(os.path.join(scan, 'images', subfolder, file))

            assert len(self.images) == len(self.depths) == len(self.labels) == len(self.instances)

            self.images.sort()
            self.depths.sort()
            self.labels.sort()
            self.instances.sort()

        else:
            print(f"Loaded {self.__class__.__name__} dataset without files")

        self.list_of_semantic_labels = np.arange(self.N_CLASSES)
        self._class_names = None
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
        color = h5py.File(self.images[idx], 'r')
        color = np.array(color['dataset'], dtype=np.float)
        color = np.clip(color, 0, 1)
        return color

    def load_depth(self, idx):
        depth = h5py.File(self.depths[idx], 'r')
        depth = np.array(depth['dataset'], dtype=np.float)
        return depth

    def load_instance(self, idx, sample):
        instance = h5py.File(self.instances[idx], 'r')
        instance = np.array(instance['dataset'])
        instance[instance == -1] = 0
        semantic_instance = sample['label'] * (instance > 0)
        return instance, semantic_instance

    def load_label(self, idx):
        label = h5py.File(self.labels[idx], 'r')
        label = np.array(label['dataset'])
        label[label == -1] = 0
        return label

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
