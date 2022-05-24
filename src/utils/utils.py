from time import perf_counter
import logging
import os, shutil
import glob
import functools
import sys
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm, trange
import math

sys.path.append('../')
import data.dataset
from data.dataset import SegmentationDataset, RegistrationDataset

log_time = False


def timer(func):
    @functools.wraps(func)
    def time_wrapper(*args, **kwargs):
        if log_time:
            tic = perf_counter()
            value = func(*args, **kwargs)
            toc = perf_counter()
            time = toc - tic
            logging.getLogger("TIME").info("Function %s takes %.5f s",
                                           func.__name__, time)
        else:
            value = func(*args, **kwargs)
        return value

    return time_wrapper


def inspect_single_data_pair(class_name, ids, dataset, transform=None):
    # load single image pair
    class_ = getattr(data.dataset, class_name)
    data_set = class_(ids, dataset=dataset, transform=transform)
    dataloader = DataLoader(data_set, batch_size=1, shuffle=True)
    x, y, _ = next(iter(dataloader))

    # get some statistics
    print(f'x = shape: {x.shape}; type: {x.dtype}')
    print(f'x = min: {x.min()}; max: {x.max()}')
    print(f'y = shape: {y.shape}; type: {y.dtype}')
    print(f'y = min: {y.min()}; max: {y.max()}')

    # plot images
    fig = plt.figure()
    fig.add_subplot(1, 2, 1)
    plt.imshow(x.view(x.shape[1], x.shape[2], 1))
    fig.add_subplot(1, 2, 2)
    plt.imshow(y.view(y.shape[1], y.shape[2], 1))
    plt.show()


def remove_folder_contents(folder_path: str):
    folder = glob.glob(folder_path + '*')
    for f in folder:
        if os.path.isfile(f) or os.path.islink(f):
            os.unlink(f)
        elif os.path.isdir(f):
            shutil.rmtree(f)
        else:
            raise AttributeError(
                f"You try to delete something which is either a file nor a folder!")
    print(f"\nRemoved all files from folder: {folder_path}")


class LearningRateFinder:
    """
    From: https://towardsdatascience.com/creating-and-training-a-u-net-model-with-pytorch-for-2d-3d-semantic-segmentation-training-3-4-8242d31de234
    Train a model using different learning rates within a range to find the optimal learning rate.
    """

    def __init__(self,
                 model: nn.Module,
                 criterion,
                 optimizer,
                 device
                 ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.loss_history = {}
        self._model_init = model.state_dict()
        self._opt_init = optimizer.state_dict()
        self.device = device

    def fit(self,
            data_loader: torch.utils.data.DataLoader,
            steps=100,
            min_lr=1e-7,
            max_lr=1,
            constant_increment=False
            ):
        """
        Trains the model for number of steps using varied learning rate and store the statistics
        """
        self.loss_history = {}
        self.model.to(self.device)
        self.model.train()
        current_lr = min_lr
        steps_counter = 0
        epochs = math.ceil(steps / len(data_loader))

        progressbar = trange(epochs, desc='Progress')
        for epoch in progressbar:
            batch_iter = tqdm(enumerate(data_loader), 'Training', total=len(data_loader),
                              leave=False)

            for i, (x, y, _) in batch_iter:
                x, y = x.to(self.device), y.to(self.device)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = current_lr
                self.optimizer.zero_grad()
                out = self.model(x)
                loss = self.criterion(out, y)
                loss.backward()
                self.optimizer.step()
                self.loss_history[current_lr] = loss.item()

                steps_counter += 1
                if steps_counter > steps:
                    break

                if constant_increment:
                    current_lr += (max_lr - min_lr) / steps
                else:
                    current_lr = current_lr * (max_lr / min_lr) ** (1 / steps)

    def plot(self,
             smoothing=True,
             clipping=True,
             smoothing_factor=0.1
             ):
        """
        Shows loss vs learning rate(log scale) in a matplotlib plot
        """
        loss_data = pd.Series(list(self.loss_history.values()))
        lr_list = list(self.loss_history.keys())
        if smoothing:
            loss_data = loss_data.ewm(alpha=smoothing_factor).mean()
            loss_data = loss_data.divide(pd.Series(
                [1 - (1.0 - smoothing_factor) ** i for i in
                 range(1, loss_data.shape[0] + 1)]))  # bias correction
        if clipping:
            loss_data = loss_data[10:-5]
            lr_list = lr_list[10:-5]
        plt.plot(lr_list, loss_data)
        plt.xscale('log')
        plt.title('Loss vs Learning rate')
        plt.xlabel('Learning rate (log scale)')
        plt.ylabel('Loss (exponential moving average)')
        plt.show()

    def reset(self):
        """
        Resets the model and optimizer to its initial state
        """
        self.model.load_state_dict(self._model_init)
        self.optimizer.load_state_dict(self._opt_init)
        print('Model and optimizer in initial state.')
