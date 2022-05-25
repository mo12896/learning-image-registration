"""DatasetHandler for lung CTs"""
import os
import sys

sys.path.append('../')

import SimpleITK as sitk
from tqdm import tqdm
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
from transforms.transforms_handler import *
import random

import systemsetup as setup


# TODO: Abstract into baseclass and METAIO class
class SegmentationDataset(Dataset):
    """
    Base class for all datasets. It implements a map-style dataset, see
    https://pytorch.org/docs/stable/data.html.
    """

    def __init__(self, ids: list, dataset: str, transform=None, use_cache=False,
                 pre_transform=None):
        """Constructor of DatasetHandler"""
        self.list_ids = ids
        self.dataset = dataset
        self.transform = transform
        self.use_cache = use_cache
        self.pre_transform = pre_transform

        # data caching, preprocessing and multiprocessing
        if self.use_cache:
            print("Cache and preprocess data in a multiprocessing context...")
            from multiprocessing import Pool
            from itertools import repeat

            indices = range(len(ids))
            with Pool() as pool:
                self.cached_data = pool.starmap(self._load_data,
                                                zip(indices, repeat(self.pre_transform)))
            print("Done!")

            # # Uncomment, if no multiprocessing is needed
            # self.cached_data = []
            # progressbar = tqdm(range(len(self.list_ids)), desc='Caching')
            # for index in progressbar:
            #     x, y, img_id = _load_data(index)
            #     self.cached_data.append((x, y, img_id))

    def __len__(self):
        """Return total number of samples"""
        return len(self.list_ids)

    def __getitem__(self, index):
        """Return single sample from dataset"""
        if isinstance(index, slice):
            raise NotImplementedError
        if isinstance(index, int):
            if index < 0 or index > len(self.list_ids):
                raise IndexError(f"Index {index} is out of range!")

            if self.use_cache:
                # TODO: change placeholder into new transform API!
                x, y, img_id = self.cached_data[index]
                x, y = x.numpy(), y.numpy()
                x, y = self.transform(x, y)
                x, y = torch.Tensor(x), torch.Tensor(y)
                return [x, y, img_id]
            else:
                x, y, img_id = self._get_item_from_index(index)
            return self.transform_data(x, y, img_id, tf=self.transform)

        raise TypeError("Invalid data type!")

    def _get_item_from_index(self, index: int):
        img_id = self.list_ids[index]
        INPUT_DIR = setup.INTERIM_DATA_DIR

        # TODO: in future, not hard-coded!
        image = img_id + '_Fixed.nii'
        x_path = os.path.join(INPUT_DIR, self.dataset + '/scans/' + image)
        y_path = os.path.join(INPUT_DIR, self.dataset + '/masks/' + image)

        x = torch.Tensor(sitk.GetArrayFromImage(sitk.ReadImage(x_path, sitk.sitkInt16)))
        y = torch.Tensor(sitk.GetArrayFromImage(sitk.ReadImage(y_path, sitk.sitkInt16)))

        return x, y, img_id

    # TODO: implement typecasting
    @staticmethod
    def transform_data(x, y, img_id, tf=None):
        if tf:
            return tf((x, y, img_id))
        else:
            raise AssertionError("No transform is called!")

    def _load_data(self, index: int, tf=None):
        x, y, img_id = self._get_item_from_index(index)
        x, y, img_id = self.transform_data(x, y, img_id, tf=tf)
        return x, y, img_id


class RegistrationDataset(Dataset):
    """
    Base class for all datasets. It implements a map-style dataset, see
    https://pytorch.org/docs/stable/data.html.
    """

    def __init__(self, ids: list, dataset: str, transform=None):
        """Constructor of DatasetHandler"""
        self.list_ids = ids
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        """Return total number of samples"""
        return len(self.list_ids)

    def __getitem__(self, index):
        """Return single sample from dataset"""
        if isinstance(index, slice):
            raise NotImplementedError
        if isinstance(index, int):
            if index < 0 or index > len(self.list_ids):
                raise IndexError(f"Index {index} is out of range!")
            return self.get_item_from_index(index)
        raise TypeError("Invalid data type!")

    def get_item_from_index(self, index: int):
        id = self.list_ids[index]
        INPUT_DIR = setup.INTERIM_DATA_DIR
        reg_modes = ['_Fixed', '_Moving']
        sample = []

        for mode in reg_modes:
            image = id + mode + '.nii'
            path = INPUT_DIR + self.dataset + '/scans/' + image
            img = sitk.ReadImage(path, sitk.sitkInt16)
            t_image = torch.Tensor(sitk.GetArrayFromImage(img))
            sample.append(t_image)
        sample.append(id)

        if self.transform:
            sample = self.transform(sample)

        return sample
