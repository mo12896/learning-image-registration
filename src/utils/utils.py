import time
import logging
import functools
import sys
from torch.utils.data import DataLoader
from data.dataset import SegmentationDataset, RegistrationDataset
import matplotlib.pyplot as plt

sys.path.append('../')
import data.dataset

log_time = False


def timer(func):
    @functools.wraps(func)
    def time_wrapper(*args, **kwargs):
        if log_time:
            tic = time.perf_counter()
            value = func(*args, **kwargs)
            toc = time.perf_counter()
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
