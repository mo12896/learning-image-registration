import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiceLoss(nn.Module):
    """
    According to: https://github.com/pytorch/pytorch/issues/1249
    """

    def __init__(self, weights=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, y_pred, y_hat, smooth=1):
        # flatten the input values
        y_pred = y_pred.view(-1)
        y_hat = y_hat.view(-1)

        intersection = (y_pred * y_hat).sum()
        # smooth value for numerical stability
        dice = (2. * intersection + smooth) / (y_pred.sum() + y_hat.sum() + smooth)
        return 1 - dice


class IoULoss(nn.Module):
    """
    According to: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook
    """

    def __init__(self, weights=None, size_average=True):
        super(IoULoss, self).__init__()

    def forward(self, y_pred, y_hat, smooth=1):
        # flatten the input values
        y_pred = y_pred.view(-1)
        y_hat = y_hat.view(-1)

        intersection = (y_pred * y_hat).sum()
        total = (y_pred + y_hat).sum()
        union = total - intersection

        iou = (intersection + smooth) / (union + smooth)
        return 1 - iou
