"""DatasetHandler for lung CTs"""
import os
import sys
sys.path.append('../')

import SimpleITK as sitk
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
import random


import systemsetup as setup



#TODO: Abstract into baseclass and METAIO class
class DatasetHandler(Dataset):
    """
    Base class for all datasets. It implements a map-style dataset, see
    https://pytorch.org/docs/stable/data.html.
    """
    def __init__(self, ids: list, root: str, transform=None):
        """Constructor of DatasetHandler"""
        self.list_ids = ids
        self.image_root = root
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
        INPUT_DIR = setup.DATA_DIR
        modes = ['_Fixed', '_Moving']
        sample = []

        for mode in modes:
            image = id + mode + '.nii'
            image_path = INPUT_DIR + self.image_root + image
            img = sitk.ReadImage(image_path, sitk.sitkInt16)
            t_image = torch.Tensor(sitk.GetArrayFromImage(img))
            sample.append(t_image)
        sample = sample.append(id)

        if self.transform:
            sample = self.transform(sample)

        return sample


class DatasetHandler(Dataset):
    """
    Base class for all datasets. It implements a map-style dataset, see
    https://pytorch.org/docs/stable/data.html.
    """
    def __init__(self, ids: list, root: str, transform=None):
        """Constructor of DatasetHandler"""
        self.list_ids = ids
        self.image_root = root
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
        INPUT_DIR = setup.DATA_DIR
        modes = ['_Fixed', '_Moving']
        images = []

        for mode in modes:
            image = id + mode + '.nii'
            image_path = INPUT_DIR + self.image_root + image
            img = sitk.ReadImage(image_path, sitk.sitkInt16)
            t_image = torch.Tensor(sitk.GetArrayFromImage(img))
            images.append(t_image)
        sample = [images[0], images[1], id]

        if self.transform:
            sample = self.transform(sample)

        return sample






