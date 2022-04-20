import os

import SimpleITK as sitk
import sys

import torch
import torch.nn.functional as F






#TODO: Define Datahandler for MetaIO-format, using simpleITK
class DatasetHandler(torch.utils.data.Dataset):

    """

    """
    def __init__(self):
        raise NotImplemented

    def __getitem__(self, item):
        raise NotImplemented

    def __len__(self):
        raise NotImplemented