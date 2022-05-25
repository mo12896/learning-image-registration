import logging
import os
import sys

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

sys.path.append('../')
import systemsetup as setup

from data.dataset import SegmentationDataset, RegistrationDataset
from utils.modes import ExeModes
from utils.logging import *
from utils.evaluate import Evaluator
from transforms.tensor_transforms import Create2D, Rescale, AddChannel, NormalizeSample
from utils.eval_metrics import DiceLoss
from utils.visualization import plot_images_in_row


def inference_pipeline(hyper: dict, log_level: str, notebook: bool, exp_name: str):
    """..."""
    # TODO: Get proper test-data (do not use validation data here!)
    train_setup = {k.lower(): v for k, v in hyper['SETUP'].items()}
    hyper = {k.lower(): v for k, v in hyper['HYPERPARAMETERS'].items()}
    random_seed = train_setup['random_seed']

    raw_data = setup.RAW_DATA_DIR + 'EMPIRE10/scans/'
    dataset = train_setup['dataset']
    ids = list(set([x.split('_')[0] for x in os.listdir(raw_data)]))
    random_seed = train_setup['random_seed']
    test_size = train_setup['split']

    """Initalize Logger"""
    init_logger(ExeModes.TEST.name, log_level, setup.LOG_DIR, mode=ExeModes.TEST)
    test_logger = logging.getLogger(ExeModes.TEST.name)
    test_logger.info("Start inference '%s'...")

    """Setup Data Generator"""
    partition = {'train': (train_test_split(
        ids, test_size=test_size, random_state=random_seed))[0], 'validation': (train_test_split(
        ids, test_size=test_size, random_state=random_seed))[1]}

    shape = (256, 256)
    pre_transform = transforms.Compose([
        Create2D('y'),
        AddChannel(axs=0),
        Rescale(shape)
    ])

    validation_set = SegmentationDataset(partition['validation'], dataset=dataset,
                                         transform=None, use_cache=True,
                                         pre_transform=pre_transform)

    validation_loader = DataLoader(validation_set, batch_size=hyper['batch_size'],
                                   shuffle=True)

    """Inference"""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet()
    model.load_state_dict(torch.load('../models/best.model'))
    model.to(device)

    eval_metric = DiceLoss()
    evaluator = Evaluator(validation_loader=validation_loader,
                          eval_metric=eval_metric,
                          device=device,
                          notebook=notebook,
                          verbose=True)

    evaluator.validate(model, 0)
