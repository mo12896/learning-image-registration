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
    def __init__(self, ids: list, dataset: str, task: str, transform=None):
        """Constructor of DatasetHandler"""
        self.list_ids = ids
        self.dataset = dataset
        self.transform = transform
        self.task = task

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
        seg_modes = ['/scans/', '/masks/']
        sample = []

        if self.task == 'REG':
            for mode in reg_modes:
                image = id + mode + '.nii'
                path = INPUT_DIR + self.dataset + '/scans/' + image
                img = sitk.ReadImage(path, sitk.sitkInt16)
                t_image = torch.Tensor(sitk.GetArrayFromImage(img))
                sample.append(t_image)
            sample.append(id)
        elif self.task == 'SEG':
            #TODO: in future, not hard-coded!
            image = id + '_Fixed.nii'
            for mode in seg_modes:
                path = os.path.join(INPUT_DIR, self.dataset + mode + image)
                img = sitk.ReadImage(path, sitk.sitkInt16)
                t_image = torch.Tensor(sitk.GetArrayFromImage(img))
                sample.append(t_image)
            sample.append(id)
        else:
            raise ValueError('Please use either REG oder SEG taks!')

        if self.transform:
            sample = self.transform(sample)

        return sample





