import os

import SimpleITK as sitk
import sys
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset
import torch.io as tio
import random




#TODO: Define Datahandler for MetaIO-format, using simpleITK
class DatasetHandler(Dataset):

    """

    """
    def __init__(self, root, data_file, patients=None, mode):
        self.root = root
        self.mode = mode
        assert os.path.isfile(data_file), 'Please provide a valid data file. \n (not valid {})'.format(data_file)
        self.data_file = data_file

        self.filenames = None
        self.labelnames = None
        self.patients = patients


    def __getitem__(self, index):
        patient_id = self.patients[index]
        fnames = self.filenames[index]
        lnames = self.labelnames[index]
        return patient_id, fnames, lnames

    def __len__(self):
        return len(self.filenames)